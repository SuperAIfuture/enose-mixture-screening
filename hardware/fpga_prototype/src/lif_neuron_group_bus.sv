// ============================================================
// 2) LIF 神经元组
// 膜电位更新：v_mem = v_mem + (result_out - v_mem) / 2
// 阈值：signed 5
// 条件：v_mem_next > THRESHOLD 时发放脉冲
// 发放后：膜电位清零
// sample_start：新样本开始时清空全部膜电位
// ============================================================
module lif_neuron_group #(
    parameter int OUT_NUM    = 7,
    parameter int DATA_WIDTH = 20,
    parameter logic signed [DATA_WIDTH-1:0] THRESHOLD = 20'sd5
)(
    input  logic                                   clk,
    input  logic                                   rst_n,
    input  logic                                   sample_start,
    input  logic                                   update_en,
    input  logic signed [DATA_WIDTH-1:0]           result_out [OUT_NUM],

    output logic [OUT_NUM-1:0]                     spike_out,
    output logic signed [DATA_WIDTH-1:0]           v_mem_out [OUT_NUM]
);

    logic signed [DATA_WIDTH-1:0] v_mem      [OUT_NUM];
    logic signed [DATA_WIDTH-1:0] delta      [OUT_NUM];
    logic signed [DATA_WIDTH-1:0] v_mem_next [OUT_NUM];

    genvar i;
    generate
        for (i = 0; i < OUT_NUM; i++) begin : GEN_LIF_COMB
            always_comb begin
                delta[i]      = (result_out[i] - v_mem[i]) >>> 1;
                v_mem_next[i] = v_mem[i] + delta[i];
            end
        end
    endgenerate

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int k = 0; k < OUT_NUM; k++) begin
                v_mem[k]     <= '0;
                v_mem_out[k] <= '0;
                spike_out[k] <= 1'b0;
            end
        end
        else if (sample_start) begin
            for (int k = 0; k < OUT_NUM; k++) begin
                v_mem[k]     <= '0;
                v_mem_out[k] <= '0;
                spike_out[k] <= 1'b0;
            end
        end
        else begin
            spike_out <= '0;

            if (update_en) begin
                for (int k = 0; k < OUT_NUM; k++) begin
                    if (v_mem_next[k] > THRESHOLD) begin
                        spike_out[k] <= 1'b1;
                        v_mem[k]     <= '0;
                        v_mem_out[k] <= '0;
                    end
                    else begin
                        spike_out[k] <= 1'b0;
                        v_mem[k]     <= v_mem_next[k];
                        v_mem_out[k] <= v_mem_next[k];
                    end
                end
            end
            else begin
                for (int k = 0; k < OUT_NUM; k++) begin
                    v_mem_out[k] <= v_mem[k];
                end
            end
        end
    end

endmodule