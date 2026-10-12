module data_loader #(
    parameter INPUT_LENGTH = 800
)(
    input                      clk,
    input                      rst_n,
    input       [7:0]          uart_rx_data,
    input                      uart_rx_done,
    output reg  [INPUT_LENGTH-1:0] sensor_data,
    output reg                 cal_start
);

// 计算需要接收的字节数
localparam BYTE_COUNT = INPUT_LENGTH / 8;

// 状态定义
localparam IDLE = 1'b0;
localparam RECV = 1'b1;

reg                    state;
reg [9:0]              byte_cnt;
reg [INPUT_LENGTH-1:0] data_buffer;

// 最终拼接完成后的原始数据
wire [INPUT_LENGTH-1:0] raw_data;

// bit翻转后的数据
wire [INPUT_LENGTH-1:0] reversed_data;

assign raw_data = {data_buffer[INPUT_LENGTH-9:0], uart_rx_data};

genvar i;
generate
    for (i = 0; i < INPUT_LENGTH; i = i + 1) begin : GEN_REVERSE
        assign reversed_data[i] = raw_data[INPUT_LENGTH-1-i];
    end
endgenerate

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state       <= IDLE;
        byte_cnt    <= 10'd0;
        data_buffer <= {INPUT_LENGTH{1'b0}};
        sensor_data <= {INPUT_LENGTH{1'b0}};
        cal_start   <= 1'b0;
    end
    else begin
        // 默认值
        cal_start   <= 1'b0;
        sensor_data <= {INPUT_LENGTH{1'b0}};

        case (state)
            IDLE: begin
                if (uart_rx_done) begin
                    // 第一个字节先放低位，后续继续按原先移位方式拼接
                    data_buffer[7:0] <= uart_rx_data;
                    byte_cnt <= 10'd1;
                    state <= RECV;
                end
            end

            RECV: begin
                if (uart_rx_done) begin
                    if (byte_cnt < BYTE_COUNT - 1) begin
                        // 保持你原来的拼接方式：先接收的逐渐移到高位
                        data_buffer <= {data_buffer[INPUT_LENGTH-9:0], uart_rx_data};
                        byte_cnt <= byte_cnt + 10'd1;
                    end
                    else begin
                        // 最后一个字节到来时，先得到完整拼接结果 raw_data
                        data_buffer <= {data_buffer[INPUT_LENGTH-9:0], uart_rx_data};

                        // 输出 bit 翻转后的结果
                        sensor_data <= reversed_data;
                        cal_start   <= 1'b1;

                        byte_cnt <= 10'd0;
                        state    <= IDLE;
                    end
                end
            end

            default: begin
                state <= IDLE;
            end
        endcase
    end
end

endmodule
