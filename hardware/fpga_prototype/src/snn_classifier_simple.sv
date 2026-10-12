// ============================================================
// 3) 最终分类模块
// 一个样本共 30 帧
// 对 7 个神经元在 30 帧中的 spike 次数计数
// 计数最多的神经元索引作为分类结果
// 并列时取索引较小者
// ============================================================
module snn_classifier #(
    parameter int OUT_NUM      = 7,
    parameter int FRAME_NUM    = 30,
    parameter int COUNT_WIDTH  = 6,
    parameter int CLASS_WIDTH  = 3
)(
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        sample_start,
    input  logic                        frame_valid,
    input  logic [OUT_NUM-1:0]          spike_out,

    output logic [CLASS_WIDTH-1:0]      class_idx,
    output logic                        class_valid,
    output logic [5:0]                  frame_cnt_out,
    output logic [COUNT_WIDTH-1:0]      spike_count [OUT_NUM]
);

    logic [5:0] frame_cnt;

    logic [COUNT_WIDTH-1:0] next_spike_count [OUT_NUM];
    logic [COUNT_WIDTH-1:0] max_count_next;
    logic [CLASS_WIDTH-1:0] max_idx_next;

    // 加上当前帧后的计数
    always_comb begin
        for (int i = 0; i < OUT_NUM; i++) begin
            next_spike_count[i] = spike_count[i];
            if (spike_out[i]) begin
                next_spike_count[i] = spike_count[i] + 1'b1;
            end
        end
    end

    // argmax
    always_comb begin
        max_idx_next   = '0;
        max_count_next = next_spike_count[0];

        for (int i = 1; i < OUT_NUM; i++) begin
            if (next_spike_count[i] > max_count_next) begin
                max_count_next = next_spike_count[i];
                max_idx_next   = i[CLASS_WIDTH-1:0];
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            frame_cnt      <= '0;
            frame_cnt_out  <= '0;
            class_idx      <= '0;
            class_valid    <= 1'b0;

            for (int i = 0; i < OUT_NUM; i++) begin
                spike_count[i] <= '0;
            end
        end
        else begin
            class_valid <= 1'b0;

            if (sample_start) begin
                frame_cnt     <= '0;
                frame_cnt_out <= '0;
                class_idx     <= '0;

                for (int i = 0; i < OUT_NUM; i++) begin
                    spike_count[i] <= '0;
                end
            end
            else if (frame_valid) begin
                for (int i = 0; i < OUT_NUM; i++) begin
                    spike_count[i] <= next_spike_count[i];
                end

                frame_cnt     <= frame_cnt + 1'b1;
                frame_cnt_out <= frame_cnt + 1'b1;

                if (frame_cnt == FRAME_NUM-1) begin
                    class_idx   <= max_idx_next;
                    class_valid <= 1'b1;
                end
            end
        end
    end

endmodule