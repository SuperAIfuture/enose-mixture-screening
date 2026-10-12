`timescale 1ns / 1ps

module top__system #(
    parameter integer INPUT_NUM   = 800,
    parameter integer OUT_NUM     = 7,
    parameter integer ROM_DEPTH   = 100,
    parameter integer ROM_WIDTH   = 64,
    parameter integer ACC_WIDTH   = 20,
    parameter integer ADDR_WIDTH  = 7,
    parameter integer FRAME_NUM   = 30,
    parameter integer COUNT_WIDTH = 6,
    parameter integer CLASS_WIDTH = 3
)(
    input                    clk,
    input                    rst_n,
    input                    uart_rxd,

    // LCD接口
    output                   lcd_hs,
    output                   lcd_vs,
    output                   lcd_de,
    inout      [23:0]        lcd_rgb,
    output                   lcd_bl,
    output                   lcd_rst,
    output                   lcd_clk
);

    //==========================================================
    // SNN分类结果信号
    //==========================================================
    wire [CLASS_WIDTH-1:0] class_idx;
    wire                   class_valid;

    //==========================================================
    // 内部 sample_start：复位释放后自动产生一个时钟脉冲
    //==========================================================
    reg rst_n_d;
    reg sample_start;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            rst_n_d       <= 1'b0;
            sample_start  <= 1'b0;
        end
        else begin
            rst_n_d      <= 1'b1;
            sample_start <= ~rst_n_d;   // 复位释放后拉高1拍
        end
    end

    //==========================================================
    // SNN顶层：UART输入 -> 分类输出
    //==========================================================
    snn_top_uart #(
        .INPUT_NUM   (INPUT_NUM),
        .OUT_NUM     (OUT_NUM),
        .ROM_DEPTH   (ROM_DEPTH),
        .ROM_WIDTH   (ROM_WIDTH),
        .ACC_WIDTH   (ACC_WIDTH),
        .ADDR_WIDTH  (ADDR_WIDTH),
        .FRAME_NUM   (FRAME_NUM),
        .COUNT_WIDTH (COUNT_WIDTH),
        .CLASS_WIDTH (CLASS_WIDTH),
        .THRESHOLD   (20'sd5)
    ) u_snn_top_uart (
        .clk         (clk),
        .rst_n       (rst_n),
        .uart_rxd    (uart_rxd),
        .sample_start(sample_start),
        .class_idx   (class_idx),
        .class_valid (class_valid)
    );

    //==========================================================
    // LCD显示顶层：根据分类结果显示不同字模
    //==========================================================
    lcd_rgb_char u_lcd_rgb_char (
        .clk    (clk),
        .rst_n  (rst_n),
        .class_idx  (class_idx),
        .class_valid(class_valid),
        .lcd_hs     (lcd_hs),
        .lcd_vs     (lcd_vs),
        .lcd_de     (lcd_de),
        .lcd_rgb    (lcd_rgb),
        .lcd_bl     (lcd_bl),
        .lcd_rst    (lcd_rst),
        .lcd_clk    (lcd_clk)
    );

endmodule
