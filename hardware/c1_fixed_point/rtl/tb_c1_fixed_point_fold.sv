`timescale 1ns/1ps
`default_nettype none

`ifndef SAMPLE_COUNT
`define SAMPLE_COUNT 1
`endif

`ifndef FOLD_INDEX
`define FOLD_INDEX 0
`endif

module tb_c1_fixed_point_fold;
    localparam integer SAMPLE_COUNT = `SAMPLE_COUNT;
    localparam integer FOLD_INDEX = `FOLD_INDEX;
    localparam integer FEATURE_COUNT = 800;
    localparam integer CLASS_COUNT = 7;
    localparam integer INPUT_WIDTH = 12;
    localparam integer ACCUMULATOR_WIDTH = 64;

    logic [INPUT_WIDTH-1:0] input_vector [0:FEATURE_COUNT-1];
    logic signed [ACCUMULATOR_WIDTH-1:0] class_score [0:CLASS_COUNT-1];
    logic [$clog2(CLASS_COUNT)-1:0] predicted_class;
    logic [INPUT_WIDTH-1:0] input_memory [0:SAMPLE_COUNT*FEATURE_COUNT-1];
    logic [2:0] expected_fixed_class [0:SAMPLE_COUNT-1];
    logic [2:0] expected_float_class [0:SAMPLE_COUNT-1];

    c1_fixed_point_linear_800x7 dut (
        .input_vector(input_vector),
        .class_score(class_score),
        .predicted_class(predicted_class)
    );

    integer sample_index;
    integer feature_index;
    integer fixed_failures;
    integer float_disagreements;

    initial begin
        $readmemh("inputs.mem", input_memory);
        $readmemh("expected_fixed_classes.mem", expected_fixed_class);
        $readmemh("expected_float_classes.mem", expected_float_class);
        fixed_failures = 0;
        float_disagreements = 0;
        for (sample_index = 0; sample_index < SAMPLE_COUNT; sample_index = sample_index + 1) begin
            for (feature_index = 0; feature_index < FEATURE_COUNT; feature_index = feature_index + 1)
                input_vector[feature_index] = input_memory[sample_index*FEATURE_COUNT+feature_index];
            #1;
            if (predicted_class !== expected_fixed_class[sample_index]) begin
                $display(
                    "FAIL fold=%0d sample=%0d fixed_class=%0d expected=%0d",
                    FOLD_INDEX,
                    sample_index,
                    predicted_class,
                    expected_fixed_class[sample_index]
                );
                fixed_failures = fixed_failures + 1;
            end
            if (predicted_class !== expected_float_class[sample_index])
                float_disagreements = float_disagreements + 1;
        end
        if ((fixed_failures == 0) && (float_disagreements == 0))
            $display(
                "C1_FIXED_POINT_RTL_PASS fold=%0d samples=%0d fixed=%0d/%0d float=%0d/%0d",
                FOLD_INDEX,
                SAMPLE_COUNT,
                SAMPLE_COUNT,
                SAMPLE_COUNT,
                SAMPLE_COUNT,
                SAMPLE_COUNT
            );
        else
            $display(
                "C1_FIXED_POINT_RTL_FAIL fold=%0d fixed_failures=%0d float_disagreements=%0d",
                FOLD_INDEX,
                fixed_failures,
                float_disagreements
            );
        $finish;
    end
endmodule

`default_nettype wire
