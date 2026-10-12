`timescale 1ns/1ps
`default_nettype none
module tb_c1_c2_bridge_350;
    localparam int SAMPLE_COUNT = 350;
    localparam int TIME_STEPS = 30;
    localparam int INPUT_FEATURES = 800;
    localparam int OUTPUT_CLASSES = 7;
    logic clk = 1'b0;
    always #5 clk = ~clk;
    logic rst_n = 1'b0;
    logic start_sample = 1'b0;
    logic frame_valid = 1'b0;
    logic [INPUT_FEATURES-1:0] frame_in = '0;
    logic busy, done;
    logic [$clog2(OUTPUT_CLASSES)-1:0] class_idx;
    logic [OUTPUT_CLASSES-1:0][15:0] output_counts;
    logic [INPUT_FEATURES-1:0] frames [0:SAMPLE_COUNT*TIME_STEPS-1];
    logic [15:0] expected_count [0:SAMPLE_COUNT*OUTPUT_CLASSES-1];
    logic [2:0] expected_class [0:SAMPLE_COUNT-1];

    initial begin
        $readmemh("bridge_frames.mem", frames);
        $readmemh("bridge_expected_counts.mem", expected_count);
        $readmemh("bridge_expected_classes.mem", expected_class);
    end

    tloc_qsnn_fixed_core dut (
        .clk(clk), .rst_n(rst_n), .start_sample(start_sample), .frame_valid(frame_valid),
        .frame_in(frame_in), .busy(busy), .done(done), .class_idx(class_idx),
        .output_counts(output_counts)
    );

    integer sample_index;
    integer frame_index;
    integer class_index;
    integer failures;
    integer sample_limit;
    initial begin
        failures = 0;
        if (!$value$plusargs("SAMPLES=%d", sample_limit))
            sample_limit = SAMPLE_COUNT;
        if ((sample_limit < 1) || (sample_limit > SAMPLE_COUNT)) begin
            $display("FAIL invalid sample limit=%0d", sample_limit);
            $finish;
        end
        repeat (3) @(posedge clk);
        rst_n <= 1'b1;
        for (sample_index = 0; sample_index < sample_limit; sample_index = sample_index + 1) begin
            @(posedge clk);
            start_sample <= 1'b1;
            @(posedge clk);
            start_sample <= 1'b0;
            for (frame_index = 0; frame_index < TIME_STEPS; frame_index = frame_index + 1) begin
                frame_in <= frames[sample_index*TIME_STEPS+frame_index];
                frame_valid <= 1'b1;
                @(posedge clk);
            end
            frame_valid <= 1'b0;
            frame_in <= '0;
            @(posedge clk);
            if (!done) begin
                $display("FAIL sample=%0d missing done", sample_index);
                failures = failures + 1;
            end
            if (class_idx !== expected_class[sample_index]) begin
                $display("FAIL sample=%0d class=%0d expected=%0d", sample_index, class_idx, expected_class[sample_index]);
                failures = failures + 1;
            end
            for (class_index = 0; class_index < OUTPUT_CLASSES; class_index = class_index + 1) begin
                if (output_counts[class_index] !== expected_count[sample_index*OUTPUT_CLASSES+class_index]) begin
                    $display("FAIL sample=%0d class=%0d count=%0d expected=%0d", sample_index,
                             class_index, output_counts[class_index],
                             expected_count[sample_index*OUTPUT_CLASSES+class_index]);
                    failures = failures + 1;
                end
            end
        end
        if (failures == 0)
            $display("C1_C2_BRIDGE_RTL_PASS samples=%0d decisions=%0d/%0d counts=%0d/%0d",
                     sample_limit, sample_limit, sample_limit, sample_limit, sample_limit);
        else
            $display("C1_C2_BRIDGE_RTL_FAIL failures=%0d", failures);
        $finish;
    end
endmodule
`default_nettype wire
