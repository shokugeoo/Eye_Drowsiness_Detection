#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "mxc.h"
#include "dma.h"
#include "camera.h"
#include "cnn.h"
#include "board.h"
#include "uart.h"

#define CAMERA_FREQ       10000000u
#define FRAME_WIDTH       160u
#define FRAME_HEIGHT      120u
#define FRAME_PIXELS      (FRAME_WIDTH * FRAME_HEIGHT)
#define FRAME_UART_BAUD   921600u
#define RGB888_DMA_BPP    4u

/* Preliminary coordinates: validate with a clear camera image. */
#define LEFT_EYE_X 65u
#define LEFT_EYE_Y 58u
#define RIGHT_EYE_X 88u
#define RIGHT_EYE_Y 58u
#define ROI_SIZE          16u
#define CNN_SIDE          24u
#define CNN_INPUT_WORDS   144u

#define CNN_INPUT_ADDR    0x50400000u
#define CNN_OUTPUT_ADDR   0x50404000u

volatile uint32_t cnn_time;

static uint32_t left_input[CNN_INPUT_WORDS];
static uint32_t right_input[CNN_INPUT_WORDS];

static uint8_t left_roi[ROI_SIZE][ROI_SIZE];
static uint8_t right_roi[ROI_SIZE][ROI_SIZE];
static uint8_t frame_gray[FRAME_PIXELS];

static uint32_t frame_count;

/* PERCLOS: rolling 60-second window, divided into 1-second buckets. */
#define PERCLOS_WINDOW_MS             60000u
#define PERCLOS_BUCKET_COUNT         60u
#define PERCLOS_MAX_SAMPLE_GAP_MS    1000u

typedef struct {
    uint32_t second_id;
    uint16_t total_ms;
    uint16_t left_closed_ms;
    uint16_t right_closed_ms;
    uint8_t valid;
} perclos_bucket_t;

static perclos_bucket_t perclos_buckets[PERCLOS_BUCKET_COUNT + 1u];

/* 1 ms timebase from the Cortex-M4 SysTick peripheral. */
static volatile uint32_t system_millis;

static uint32_t perclos_last_ms;
static uint8_t perclos_initialized;
static uint8_t perclos_last_left_closed;
static uint8_t perclos_last_right_closed;

void SysTick_Handler(void)
{
    system_millis++;
}

#if (LEFT_EYE_X + ROI_SIZE > FRAME_WIDTH) || \
    (RIGHT_EYE_X + ROI_SIZE > FRAME_WIDTH) || \
    (LEFT_EYE_Y + ROI_SIZE > FRAME_HEIGHT) || \
    (RIGHT_EYE_Y + ROI_SIZE > FRAME_HEIGHT)
#error "Eye crop lies outside the camera frame"
#endif

static void fatal_error(const char *message)
{
    printf("\nFATAL: %s\n", message);
    while (1) {
        MXC_Delay(500000);
    }
}

/* Match the RGB888 channel interpretation used by ImgCapture's converter.
 * In DMA mode, RGB888 occupies four bytes per pixel; byte 3 is padding.
 */
static uint8_t rgb888_to_gray(const uint8_t *pixel)
{
    uint32_t r = pixel[0];
    uint32_t g = pixel[1];
    uint32_t b = pixel[2];

    return (uint8_t)((77u * r + 150u * g + 29u * b + 128u) >> 8);
}

static void process_camera_row(const uint8_t *row, uint32_t frame_y)
{
    if (frame_y < FRAME_HEIGHT) {
        uint32_t offset = frame_y * FRAME_WIDTH;
        for (uint32_t x = 0; x < FRAME_WIDTH; x++) {
            frame_gray[offset + x] =
                rgb888_to_gray(&row[x * RGB888_DMA_BPP]);
        }
    }

    if (frame_y >= LEFT_EYE_Y &&
        frame_y < LEFT_EYE_Y + ROI_SIZE) {

        uint32_t y = frame_y - LEFT_EYE_Y;

        for (uint32_t x = 0; x < ROI_SIZE; x++) {
            uint32_t source_x = LEFT_EYE_X + x;
            left_roi[y][x] =
                rgb888_to_gray(&row[source_x * RGB888_DMA_BPP]);
        }
    }

    if (frame_y >= RIGHT_EYE_Y &&
        frame_y < RIGHT_EYE_Y + ROI_SIZE) {

        uint32_t y = frame_y - RIGHT_EYE_Y;

        for (uint32_t x = 0; x < ROI_SIZE; x++) {
            uint32_t source_x = RIGHT_EYE_X + x;
            right_roi[y][x] =
                rgb888_to_gray(&row[source_x * RGB888_DMA_BPP]);
        }
    }
}

static uint32_t resize_coord_q8(uint32_t output_index)
{
    int32_t q =
        (int32_t)(((2u * output_index + 1u) *
                   ROI_SIZE * 256u) /
                  (2u * CNN_SIDE)) - 128;

    if (q < 0) {
        q = 0;
    }

    const int32_t maximum = (ROI_SIZE - 1u) * 256u;
    if (q > maximum) {
        q = maximum;
    }

    return (uint32_t)q;
}

/* Matches ai8x.normalize(): round((gray/255 - 0.5)*256), clamp [-128,127]. */
static int8_t normalize_gray(uint8_t gray)
{
    int32_t value = ((int32_t)gray * 256 + 127) / 255 - 128;

    if (value > 127) value = 127;
    if (value < -128) value = -128;

    return (int8_t)value;
}

static void prepare_cnn_input(
    const uint8_t source[ROI_SIZE][ROI_SIZE],
    uint32_t output_words[CNN_INPUT_WORDS])
{
    int8_t *output = (int8_t *)output_words;

    /* Rotate the raw camera eye crop 180 degrees before resizing. */
    uint8_t rotated_roi[ROI_SIZE][ROI_SIZE];

    for (uint32_t y = 0; y < ROI_SIZE; y++) {
        for (uint32_t x = 0; x < ROI_SIZE; x++) {
            rotated_roi[y][x] =
                source[ROI_SIZE - 1u - y][ROI_SIZE - 1u - x];
        }
    }


    for (uint32_t y = 0; y < CNN_SIDE; y++) {
        uint32_t yq = resize_coord_q8(y);
        uint32_t y0 = yq >> 8;
        uint32_t y1 = (y0 + 1u < ROI_SIZE) ? y0 + 1u : y0;
        uint32_t fy = yq & 0xFFu;

        for (uint32_t x = 0; x < CNN_SIDE; x++) {
            uint32_t xq = resize_coord_q8(x);
            uint32_t x0 = xq >> 8;
            uint32_t x1 = (x0 + 1u < ROI_SIZE) ? x0 + 1u : x0;
            uint32_t fx = xq & 0xFFu;

            uint32_t p00 = rotated_roi[y0][x0];
            uint32_t p01 = rotated_roi[y0][x1];
            uint32_t p10 = rotated_roi[y1][x0];
            uint32_t p11 = rotated_roi[y1][x1];

            uint32_t top = p00 * (256u - fx) + p01 * fx;
            uint32_t bottom = p10 * (256u - fx) + p11 * fx;

            uint32_t gray =
                (top * (256u - fy) + bottom * fy + 32768u) >> 16;

            output[y * CNN_SIDE + x] =
                normalize_gray((uint8_t)gray);
        }
    }
}

static int consume_camera_rows(uint32_t *row_count)
{
    uint8_t *row;

    while ((row = get_camera_stream_buffer()) != NULL) {
        if (*row_count >= FRAME_HEIGHT) {
            release_camera_stream_buffer();
            return -1;
        }

        process_camera_row(row, *row_count);
        (*row_count)++;
        release_camera_stream_buffer();
    }

    return 0;
}

static int capture_eye_regions(void)
{
    uint32_t row_count = 0;
    uint32_t timeout = 0;

    stream_stat_t *stats = get_camera_stream_statistic();
    uint32_t overflow_before = stats->overflow_count;

    memset(left_roi, 0, sizeof(left_roi));
    memset(right_roi, 0, sizeof(right_roi));

    if (camera_start_capture_image() != STATUS_OK) {
        printf("Could not start camera capture.\n");
        return -1;
    }

    while (!camera_is_image_rcv()) {
        if (consume_camera_rows(&row_count) != 0) {
            printf("Unexpected camera row count.\n");
            return -1;
        }

        if (++timeout >= 100000000u) {
            printf("Camera capture timed out.\n");
            return -1;
        }
    }

    /* Drain any final completed row. */
    for (uint32_t settle = 0;
         row_count < FRAME_HEIGHT && settle < 100000u;
         settle++) {
        if (consume_camera_rows(&row_count) != 0) {
            return -1;
        }
    }

    uint32_t transfers = stats->dma_transfer_count;
    uint32_t overflow_delta = stats->overflow_count - overflow_before;

    if (row_count != FRAME_HEIGHT ||
        transfers != FRAME_HEIGHT ||
        overflow_delta != 0u) {
        printf("Frame rejected: incomplete or overflowing capture.\n");
        return -1;
    }

    return 0;
}

static int run_eye_inference(
    const uint32_t input[CNN_INPUT_WORDS],
    int8_t *closed_score,
    int8_t *open_score)
{
    memcpy32((uint32_t *)CNN_INPUT_ADDR, input, CNN_INPUT_WORDS);

    if (cnn_start() != CNN_OK) {
        return -1;
    }

    while (cnn_time == 0) {
        MXC_LP_EnterSleepMode();
    }

    uint32_t packed =
        *((volatile uint32_t *)CNN_OUTPUT_ADDR);

    *closed_score = (int8_t)(packed & 0xFFu);
    *open_score = (int8_t)((packed >> 8) & 0xFFu);

    return 0;
}

static const char *eye_state(int8_t closed_score, int8_t open_score)
{
    return (closed_score >= open_score) ? "CLOSED" : "OPEN";
}


/* Blocking transmission that waits for space in the UART TX FIFO. */
static void uart_write_all(const uint8_t *data, uint32_t length)
{
    mxc_uart_regs_t *uart = MXC_UART_GET_UART(CONSOLE_UART);

    while (length > 0u) {
        unsigned int written = MXC_UART_WriteTXFIFO(
            uart, data, length
        );

        if (written > 0u) {
            data += written;
            length -= written;
        }
    }
}

static uint32_t crc32_update(
    uint32_t crc,
    const uint8_t *data,
    uint32_t length)
{
    for (uint32_t i = 0; i < length; i++) {
        crc ^= data[i];

        for (uint32_t bit = 0; bit < 8u; bit++) {
            crc = (crc & 1u)
                ? ((crc >> 1) ^ 0xEDB88320u)
                : (crc >> 1);
        }
    }

    return crc;
}


/*
 * Add an observed state interval to the appropriate one-second buckets.
 * Long gaps are discarded so an old prediction is not treated as current.
 */
static void perclos_add_interval(
    uint32_t start_ms,
    uint32_t duration_ms,
    uint8_t left_closed,
    uint8_t right_closed)
{
    if (duration_ms == 0u ||
        duration_ms > PERCLOS_MAX_SAMPLE_GAP_MS) {
        return;
    }

    while (duration_ms > 0u) {
        uint32_t second_id = start_ms / 1000u;
        uint32_t offset_ms = start_ms % 1000u;
        uint32_t segment_ms = 1000u - offset_ms;

        if (segment_ms > duration_ms) {
            segment_ms = duration_ms;
        }

        perclos_bucket_t *bucket =
            &perclos_buckets[second_id % (PERCLOS_BUCKET_COUNT + 1u)];

        if (!bucket->valid || bucket->second_id != second_id) {
            bucket->second_id = second_id;
            bucket->total_ms = 0u;
            bucket->left_closed_ms = 0u;
            bucket->right_closed_ms = 0u;
            bucket->valid = 1u;
        }

        bucket->total_ms =
            (uint16_t)(bucket->total_ms + segment_ms);

        if (left_closed) {
            bucket->left_closed_ms =
                (uint16_t)(bucket->left_closed_ms + segment_ms);
        }

        if (right_closed) {
            bucket->right_closed_ms =
                (uint16_t)(bucket->right_closed_ms + segment_ms);
        }

        start_ms += segment_ms;
        duration_ms -= segment_ms;
    }
}


/* Calculate each eye's closed-time percentage over the rolling window.
 * Values are returned in tenths of a percent: 523 means 52.3%.
 */
static void perclos_calculate(
    uint32_t now_ms,
    uint16_t *left_perclos_x10,
    uint16_t *right_perclos_x10,
    uint16_t *coverage_ms)
{
    uint32_t current_second = now_ms / 1000u;
    uint32_t total_ms = 0u;
    uint32_t left_closed_ms = 0u;
    uint32_t right_closed_ms = 0u;

    for (uint32_t i = 0u; i < (PERCLOS_BUCKET_COUNT + 1u); i++) {
        perclos_bucket_t *bucket = &perclos_buckets[i];

        if (!bucket->valid) {
            continue;
        }

        uint32_t age_seconds = current_second - bucket->second_id;

        if (age_seconds >= 1u && age_seconds <= PERCLOS_BUCKET_COUNT) {
            total_ms += bucket->total_ms;
            left_closed_ms += bucket->left_closed_ms;
            right_closed_ms += bucket->right_closed_ms;
        }
    }

    *left_perclos_x10 = 0u;
    *right_perclos_x10 = 0u;
    *coverage_ms = (uint16_t)total_ms;

    if (total_ms > 0u) {
        *left_perclos_x10 = (uint16_t)(
            (left_closed_ms * 1000u + total_ms / 2u) / total_ms
        );

        *right_perclos_x10 = (uint16_t)(
            (right_closed_ms * 1000u + total_ms / 2u) / total_ms
        );
    }
}


/* Attribute elapsed time to the previous eye prediction, then update it. */
static void perclos_update(
    uint32_t now_ms,
    uint8_t left_closed_now,
    uint8_t right_closed_now,
    uint16_t *left_perclos_x10,
    uint16_t *right_perclos_x10,
    uint16_t *coverage_ms)
{
    if (!perclos_initialized) {
        perclos_last_ms = now_ms;
        perclos_last_left_closed = left_closed_now;
        perclos_last_right_closed = right_closed_now;
        perclos_initialized = 1u;
    } else {
        uint32_t elapsed_ms = now_ms - perclos_last_ms;

        perclos_add_interval(
            perclos_last_ms,
            elapsed_ms,
            perclos_last_left_closed,
            perclos_last_right_closed
        );

        perclos_last_ms = now_ms;
        perclos_last_left_closed = left_closed_now;
        perclos_last_right_closed = right_closed_now;
    }

    perclos_calculate(
        now_ms,
        left_perclos_x10,
        right_perclos_x10,
        coverage_ms
    );
}

/*
 * Packet:
 *   4 bytes  "EY24"
 *   1 byte   protocol version
 *   4 bytes  signed scores: left-closed, left-open,
 *                              right-closed, right-open
 *   4 bytes  frame number, little-endian
 *   19200    grayscale image pixels, row-major
 *   4 bytes  CRC32, little-endian
 */
/*
 * EY24 protocol version 2 packet:
 *   4 bytes  "EY24"
 *   1 byte   protocol version (2)
 *   4 bytes  signed CNN scores:
 *              left-closed, left-open, right-closed, right-open
 *   4 bytes  frame number, little-endian
 *   2 bytes  left PERCLOS x10, little-endian
 *   2 bytes  right PERCLOS x10, little-endian
 *   2 bytes  valid-time coverage in milliseconds, little-endian
 *   19200    grayscale image pixels, row-major
 *   4 bytes  CRC32, little-endian
 */
static void send_frame_packet(
    uint32_t number,
    int8_t lc, int8_t lo,
    int8_t rc, int8_t ro,
    uint16_t left_perclos_x10,
    uint16_t right_perclos_x10,
    uint16_t coverage_ms)
{
    uint8_t header[19] = {
        'E', 'Y', '2', '4',
        2u,

        (uint8_t)lc,
        (uint8_t)lo,
        (uint8_t)rc,
        (uint8_t)ro,

        (uint8_t)number,
        (uint8_t)(number >> 8),
        (uint8_t)(number >> 16),
        (uint8_t)(number >> 24),

        (uint8_t)left_perclos_x10,
        (uint8_t)(left_perclos_x10 >> 8),

        (uint8_t)right_perclos_x10,
        (uint8_t)(right_perclos_x10 >> 8),

        (uint8_t)coverage_ms,
        (uint8_t)(coverage_ms >> 8)
    };

    uint32_t crc = 0xFFFFFFFFu;
    crc = crc32_update(crc, &header[4], 15u);
    crc = crc32_update(crc, frame_gray, FRAME_PIXELS);
    crc = ~crc;

    uint8_t crc_bytes[4] = {
        (uint8_t)crc,
        (uint8_t)(crc >> 8),
        (uint8_t)(crc >> 16),
        (uint8_t)(crc >> 24)
    };

    uart_write_all(header, sizeof(header));
    uart_write_all(frame_gray, FRAME_PIXELS);
    uart_write_all(crc_bytes, sizeof(crc_bytes));
}

int main(void)
{
    int camera_id = 0;

    MXC_ICC_Enable(MXC_ICC0);
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    /* Start a 1 ms timebase for PERCLOS measurement. */
    if (SysTick_Config(SystemCoreClock / 1000u) != 0u) {
        while (1) {
        }
    }

    /* Initialize FTHR hardware, UART0 console, and CNN power supply. */
    if (Board_Init() != E_NO_ERROR) {
        while (1) {
            MXC_Delay(500000);
        }
    }

    if (Console_Shutdown() != E_NO_ERROR ||
        MXC_UART_Init(
            MXC_UART_GET_UART(CONSOLE_UART),
            FRAME_UART_BAUD,
            MXC_UART_IBRO_CLK) != E_NO_ERROR) {
        while (1) {
            MXC_Delay(500000);
        }
    }

    printf("\n=== EYE24 LIVE CAMERA INFERENCE ===\n");

    printf("Initializing DMA...\n");
    MXC_DMA_Init();

    int dma_channel = MXC_DMA_AcquireChannel();
    if (dma_channel < 0) {
        fatal_error("Could not acquire DMA channel");
    }

    printf("Initializing OV7692...\n");
    if (camera_init(CAMERA_FREQ) != STATUS_OK) {
        fatal_error("Camera initialization failed");
    }

    if (camera_get_manufacture_id(&camera_id) != STATUS_OK) {
        fatal_error("Could not read camera ID");
    }

    printf("Camera ID: 0x%04X\n", camera_id);

    /* camera_init loads the default 0xD6 orientation register.
       These calls restore the upright orientation tested during calibration.
     */
    if (camera_set_hmirror(0) != STATUS_OK ||
        camera_set_vflip(0) != STATUS_OK) {
        fatal_error("Could not configure camera orientation");
    }

    printf("Camera orientation: both flips disabled\n");
    printf("Configuring 160x120 RGB888 streaming...\n");

    if (camera_setup(
            FRAME_WIDTH,
            FRAME_HEIGHT,
            PIXFORMAT_RGB888,
            FIFO_THREE_BYTE,
            STREAMING_DMA,
            dma_channel) != STATUS_OK) {
        fatal_error("Camera setup failed");
    }

    int row_size = camera_get_stream_buffer_size();
    printf("RGB888 DMA row buffer: %d bytes\n", row_size);

    if (row_size != (int)(FRAME_WIDTH * RGB888_DMA_BPP)) {
        fatal_error("Unexpected RGB888 DMA row-buffer size");
    }

    printf("Left ROI:  (%u,%u) %ux%u\n",
           LEFT_EYE_X, LEFT_EYE_Y, ROI_SIZE, ROI_SIZE);
    printf("Right ROI: (%u,%u) %ux%u\n",
           RIGHT_EYE_X, RIGHT_EYE_Y, ROI_SIZE, ROI_SIZE);

    printf("Initializing MAX78000 CNN...\n");

    if (cnn_enable(
            MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK,
            MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1) != CNN_OK) {
        fatal_error("CNN enable failed");
    }

    if (cnn_init() != CNN_OK ||
        cnn_load_weights() != CNN_OK ||
        cnn_load_bias() != CNN_OK ||
        cnn_configure() != CNN_OK) {
        fatal_error("CNN initialization failed");
    }

    printf("CNN ready. Starting continuous inference.\n");
    printf("Eye crop positions and orientation calibrated.\\n");
    printf("PERCLOS uses a 60-second rolling window.\\n");
    printf("PERCLOS is based on binary OPEN/CLOSED predictions.\\n\\n");

    while (1) {
        int8_t lc, lo, rc, ro;

        if (capture_eye_regions() != 0) {
            printf("Frame capture failed. Retrying...\n\n");
            MXC_Delay(100000);
            continue;
        }

        prepare_cnn_input(left_roi, left_input);
        prepare_cnn_input(right_roi, right_input);

        if (run_eye_inference(left_input, &lc, &lo) != 0) {
            fatal_error("Left-eye inference failed");
        }

        if (run_eye_inference(right_input, &rc, &ro) != 0) {
            fatal_error("Right-eye inference failed");
        }

        uint16_t left_perclos_x10;
        uint16_t right_perclos_x10;
        uint16_t coverage_ms;

        uint32_t now_ms = system_millis;

        perclos_update(
            now_ms,
            (lc >= lo) ? 1u : 0u,
            (rc >= ro) ? 1u : 0u,
            &left_perclos_x10,
            &right_perclos_x10,
            &coverage_ms
        );

        frame_count++;

        send_frame_packet(
            frame_count,
            lc, lo, rc, ro,
            left_perclos_x10,
            right_perclos_x10,
            coverage_ms
        );

        MXC_Delay(100000);
    }
}
