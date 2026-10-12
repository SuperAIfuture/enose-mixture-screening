`timescale 1ns/1ps

// ============================================================
// 1) 线性层
// 功能：
//   - 输入 800 维脉冲 sensor_data
//   - 7 个输出神经元，对应 7 个 ROM
//   - 每个 ROM：100 深度，64bit 宽，每个 64bit 含 8 个 signed[7:0] 权重
//   - 对 sensor_data 中为 1 的位置，对应权重进行累加
//   - 输出 7 个神经元的加权和 result_out[0:6]
// ============================================================
module snn_linear_layer_sv #(
    parameter int INPUT_NUM  = 800,
    parameter int OUT_NUM    = 7,
    parameter int ROM_DEPTH  = 100,
    parameter int ROM_WIDTH  = 64,
    parameter int ACC_WIDTH  = 20,
    parameter int ADDR_WIDTH = 7
)(
    input  logic                                clk,
    input  logic                                rst_n,
    input  logic                                cal_start,
    input  logic [INPUT_NUM-1:0]                sensor_data,

    output logic signed [ACC_WIDTH-1:0]         result_out [OUT_NUM],
    output logic                                result_valid,
    output logic                                busy
);

    typedef enum logic [1:0] {
        IDLE  = 2'd0,
        READ  = 2'd1,
        ACCUM = 2'd2,
        DONE  = 2'd3
    } state_t;

    state_t state, next_state;

    logic [ADDR_WIDTH-1:0] rom_addr;
    logic                  rom_ena;
    logic [ROM_WIDTH-1:0]  rom_dout [OUT_NUM];

    logic [INPUT_NUM-1:0]  sensor_data_reg;
    logic [ADDR_WIDTH-1:0] block_cnt;

    logic signed [ACC_WIDTH-1:0] acc_sum     [OUT_NUM];
    logic signed [ACC_WIDTH-1:0] partial_sum [OUT_NUM];

    logic [7:0] sensor_block;

    // 取当前 block 对应的 8 个输入脉冲
    always_comb begin
        for (int k = 0; k < 8; k++) begin
            sensor_block[k] = sensor_data_reg[block_cnt*8 + k];
        end
    end

    // 8bit signed -> ACC_WIDTH signed 扩展
    function automatic logic signed [ACC_WIDTH-1:0] ext_weight(
        input logic signed [7:0] w
    );
        ext_weight = {{(ACC_WIDTH-8){w[7]}}, w};
    endfunction

    // -------------------------
    // 7 个 ROM IP 实例
    // -------------------------
    blk_mem_gen_0 u_rom_0 (
        .clka  (clk),
        .ena   (rom_ena),
        .addra (rom_addr),
        .douta (rom_dout[0])
    );

    blk_mem_gen_1 u_rom_1 (
        .clka  (clk),
        .ena   (rom_ena),
        .addra (rom_addr),
        .douta (rom_dout[1])
    );

    blk_mem_gen_2 u_rom_2 (
        .clka  (clk),
        .ena   (rom_ena),
        .addra (rom_addr),
        .douta (rom_dout[2])
    );

    blk_mem_gen_3 u_rom_3 (
        .clka  (clk),
        .ena   (rom_ena),
        .addra (rom_addr),
        .douta (rom_dout[3])
    );

    blk_mem_gen_4 u_rom_4 (
        .clka  (clk),
        .ena   (rom_ena),
        .addra (rom_addr),
        .douta (rom_dout[4])
    );

    blk_mem_gen_5 u_rom_5 (
        .clka  (clk),
        .ena   (rom_ena),
        .addra (rom_addr),
        .douta (rom_dout[5])
    );

    blk_mem_gen_6 u_rom_6 (
        .clka  (clk),
        .ena   (rom_ena),
        .addra (rom_addr),
        .douta (rom_dout[6])
    );

    // 当前 64bit 中的 8 个权重求部分和
    genvar n;
    generate
        for (n = 0; n < OUT_NUM; n++) begin : GEN_PARTIAL_SUM
            logic signed [7:0] w [8];

            assign w[0] = rom_dout[n][7:0];
            assign w[1] = rom_dout[n][15:8];
            assign w[2] = rom_dout[n][23:16];
            assign w[3] = rom_dout[n][31:24];
            assign w[4] = rom_dout[n][39:32];
            assign w[5] = rom_dout[n][47:40];
            assign w[6] = rom_dout[n][55:48];
            assign w[7] = rom_dout[n][63:56];

            always_comb begin
                partial_sum[n] = '0;
                for (int k = 0; k < 8; k++) begin
                    if (sensor_block[k]) begin
                        partial_sum[n] += ext_weight(w[k]);
                    end
                end
            end
        end
    endgenerate

    // 状态跳转
    always_comb begin
        next_state = state;
        unique case (state)
            IDLE: begin
                if (cal_start)
                    next_state = READ;
            end
            READ: begin
                // 同步 ROM：本拍给地址，下拍数据用于 ACCUM
                next_state = ACCUM;
            end
            ACCUM: begin
                if (block_cnt == ROM_DEPTH-1)
                    next_state = DONE;
                else
                    next_state = READ;
            end
            DONE: begin
                next_state = IDLE;
            end
            default: begin
                next_state = IDLE;
            end
        endcase
    end

    // ROM 使能
    assign rom_ena = (state == READ) || (state == ACCUM);

    // 时序逻辑
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= IDLE;
            rom_addr        <= '0;
            block_cnt       <= '0;
            sensor_data_reg <= '0;
            result_valid    <= 1'b0;
            busy            <= 1'b0;

            for (int i = 0; i < OUT_NUM; i++) begin
                acc_sum[i]    <= '0;
                result_out[i] <= '0;
            end
        end
        else begin
            state        <= next_state;
            result_valid <= 1'b0;

            unique case (state)
                IDLE: begin
                    busy      <= 1'b0;
                    rom_addr  <= '0;
                    block_cnt <= '0;

                    for (int i = 0; i < OUT_NUM; i++) begin
                        acc_sum[i] <= '0;
                    end

                    if (cal_start) begin
                        busy            <= 1'b1;
                        sensor_data_reg <= sensor_data;
                        rom_addr        <= '0;
                    end
                end

                READ: begin
                    busy <= 1'b1;
                end

                ACCUM: begin
                    busy <= 1'b1;

                    for (int i = 0; i < OUT_NUM; i++) begin
                        acc_sum[i] <= acc_sum[i] + partial_sum[i];
                    end

                    if (block_cnt != ROM_DEPTH-1) begin
                        block_cnt <= block_cnt + 1'b1;
                        rom_addr  <= block_cnt + 1'b1;
                    end
                end

                DONE: begin
                    busy         <= 1'b0;
                    result_valid <= 1'b1;

                    for (int i = 0; i < OUT_NUM; i++) begin
                        result_out[i] <= acc_sum[i];
                    end
                end

                default: begin
                    busy <= 1'b0;
                end
            endcase
        end
    end

endmodule