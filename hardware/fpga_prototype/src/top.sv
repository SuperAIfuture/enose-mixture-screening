`timescale 1ns / 1ps

module snn_top_uart #(
    parameter int INPUT_NUM   = 800,
    parameter int OUT_NUM     = 7,
    parameter int ROM_DEPTH   = 100,
    parameter int ROM_WIDTH   = 64,
    parameter int ACC_WIDTH   = 20,
    parameter int ADDR_WIDTH  = 7,
    parameter int FRAME_NUM   = 30,
    parameter int COUNT_WIDTH = 6,
    parameter int CLASS_WIDTH = 3,
    parameter logic signed [ACC_WIDTH-1:0] THRESHOLD = 20'sd5
)(
    input  logic                                clk,
    input  logic                                rst_n,
    input  logic                                uart_rxd,
    input  logic                                sample_start,

    output logic [CLASS_WIDTH-1:0]              class_idx,
    output logic                                class_valid
);

    logic [7:0]                          uart_rx_data;
    logic                                uart_rx_done;

    logic [INPUT_NUM-1:0]                sensor_data;
    logic                                cal_start;

    logic                                linear_busy;
    logic                                linear_result_valid;
    logic [OUT_NUM-1:0]                  spike_out;
    logic signed [ACC_WIDTH-1:0]         linear_result [OUT_NUM];
    logic signed [ACC_WIDTH-1:0]         v_mem_out [OUT_NUM];
    logic [5:0]                          frame_cnt_out;
    logic [COUNT_WIDTH-1:0]              spike_count [OUT_NUM];

    logic                                spike_valid_d;

    // UART接收
    uart_rx u_uart_rx (
        .clk         (clk),
        .rst_n       (rst_n),
        .uart_rxd    (uart_rxd),
        .uart_rx_done(uart_rx_done),
        .uart_rx_data(uart_rx_data)
    );

    // 数据拼接
    data_loader #(
        .INPUT_LENGTH(INPUT_NUM)
    ) u_data_loader (
        .clk         (clk),
        .rst_n       (rst_n),
        .uart_rx_data(uart_rx_data),
        .uart_rx_done(uart_rx_done),
        .sensor_data (sensor_data),
        .cal_start   (cal_start)
    );

    // 线性层
    snn_linear_layer_sv #(
        .INPUT_NUM (INPUT_NUM),
        .OUT_NUM   (OUT_NUM),
        .ROM_DEPTH (ROM_DEPTH),
        .ROM_WIDTH (ROM_WIDTH),
        .ACC_WIDTH (ACC_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_snn_linear_layer_sv (
        .clk         (clk),
        .rst_n       (rst_n),
        .cal_start   (cal_start),
        .sensor_data (sensor_data),
        .result_out  (linear_result),
        .result_valid(linear_result_valid),
        .busy        (linear_busy)
    );

    // LIF神经元组
    lif_neuron_group #(
        .OUT_NUM   (OUT_NUM),
        .DATA_WIDTH(ACC_WIDTH),
        .THRESHOLD (THRESHOLD)
    ) u_lif_neuron_group (
        .clk         (clk),
        .rst_n       (rst_n),
        .sample_start(sample_start),
        .update_en   (linear_result_valid),
        .result_out  (linear_result),
        .spike_out   (spike_out),
        .v_mem_out   (v_mem_out)
    );

    // spike有效延迟一拍
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            spike_valid_d <= 1'b0;
        else
            spike_valid_d <= linear_result_valid;
    end

    // 分类器
    snn_classifier #(
        .OUT_NUM     (OUT_NUM),
        .FRAME_NUM   (FRAME_NUM),
        .COUNT_WIDTH (COUNT_WIDTH),
        .CLASS_WIDTH (CLASS_WIDTH)
    ) u_snn_classifier (
        .clk          (clk),
        .rst_n        (rst_n),
        .sample_start (sample_start),
        .frame_valid  (spike_valid_d),
        .spike_out    (spike_out),
        .class_idx    (class_idx),
        .class_valid  (class_valid),
        .frame_cnt_out(frame_cnt_out),
        .spike_count  (spike_count)
    );


endmodule
