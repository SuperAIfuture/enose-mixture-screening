`timescale 1ns / 1ps

module lcd_rgb_char(
    input              clk,
    input              rst_n,
    input      [2:0]   class_idx,
    input              class_valid,

    output             lcd_hs,
    output             lcd_vs,
    output             lcd_de,
    inout      [23:0]  lcd_rgb,
    output             lcd_bl,
    output             lcd_rst,
    output             lcd_clk
);

wire  [15:0]  lcd_id;
wire          lcd_pclk;
wire  [10:0]  pixel_xpos;
wire  [10:0]  pixel_ypos;
wire  [10:0]  h_disp;
wire  [10:0]  v_disp;
wire  [23:0]  pixel_data;
wire  [23:0]  lcd_rgb_o;
wire  [23:0]  lcd_rgb_i;

reg   [2:0]   class_idx_latched;
reg           class_done;

assign lcd_rgb   = lcd_de ? lcd_rgb_o : {24{1'bz}};
assign lcd_rgb_i = lcd_rgb;

// 锁存分类结果
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        class_idx_latched <= 3'd0;
        class_done        <= 1'b0;
    end
    else if(class_valid) begin
        class_idx_latched <= class_idx;
        class_done        <= 1'b1;
    end
end

rd_id u_rd_id(
    .clk    (clk),
    .rst_n  (rst_n),
    .lcd_rgb(lcd_rgb_i),
    .lcd_id (lcd_id)
);

clk_div u_clk_div(
    .clk     (clk),
    .rst_n   (rst_n),
    .lcd_id  (lcd_id),
    .lcd_pclk(lcd_pclk)
);

lcd_display u_lcd_display(
    .lcd_pclk   (lcd_pclk),
    .rst_n      (rst_n),
    .class_idx  (class_idx_latched),
    .class_valid(class_done),
    .pixel_xpos (pixel_xpos),
    .pixel_ypos (pixel_ypos),
    .pixel_data (pixel_data)
);

lcd_driver u_lcd_driver(
    .lcd_pclk   (lcd_pclk),
    .rst_n      (rst_n),
    .lcd_id     (lcd_id),
    .pixel_data (pixel_data),
    .pixel_xpos (pixel_xpos),
    .pixel_ypos (pixel_ypos),
    .h_disp     (h_disp),
    .v_disp     (v_disp),
    .lcd_de     (lcd_de),
    .lcd_hs     (lcd_hs),
    .lcd_vs     (lcd_vs),
    .lcd_bl     (lcd_bl),
    .lcd_clk    (lcd_clk),
    .lcd_rst    (lcd_rst),
    .lcd_rgb    (lcd_rgb_o)
);

endmodule
