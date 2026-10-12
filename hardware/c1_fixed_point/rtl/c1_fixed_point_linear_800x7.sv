`timescale 1ns/1ps
`default_nettype none

// Combinational multi-bit fixed-point reference for one exported C1 outer fold.
// Run from a fold directory so weights.mem and biases.mem resolve locally.
module c1_fixed_point_linear_800x7 #(
    parameter integer FEATURE_COUNT = 800,
    parameter integer CLASS_COUNT = 7,
    parameter integer INPUT_WIDTH = 12,
    parameter integer WEIGHT_WIDTH = 24,
    parameter integer ACCUMULATOR_WIDTH = 64
)(
    input  logic [INPUT_WIDTH-1:0] input_vector [0:FEATURE_COUNT-1],
    output logic signed [ACCUMULATOR_WIDTH-1:0] class_score [0:CLASS_COUNT-1],
    output logic [$clog2(CLASS_COUNT)-1:0] predicted_class
);

    localparam integer WEIGHT_COUNT = FEATURE_COUNT * CLASS_COUNT;
    logic signed [WEIGHT_WIDTH-1:0] weight_memory [0:WEIGHT_COUNT-1];
    logic signed [ACCUMULATOR_WIDTH-1:0] bias_memory [0:CLASS_COUNT-1];
    integer class_index;
    integer feature_index;

    initial begin
        $readmemh("weights.mem", weight_memory);
        $readmemh("biases.mem", bias_memory);
    end

    always_comb begin
        for (class_index = 0; class_index < CLASS_COUNT; class_index = class_index + 1) begin
            class_score[class_index] = bias_memory[class_index];
            for (feature_index = 0; feature_index < FEATURE_COUNT; feature_index = feature_index + 1) begin
                class_score[class_index] = class_score[class_index]
                    + $signed({1'b0, input_vector[feature_index]})
                    * $signed(weight_memory[class_index*FEATURE_COUNT+feature_index]);
            end
        end
        predicted_class = '0;
        for (class_index = 1; class_index < CLASS_COUNT; class_index = class_index + 1) begin
            if (class_score[class_index] > class_score[predicted_class])
                predicted_class = class_index[$clog2(CLASS_COUNT)-1:0];
        end
    end

endmodule

`default_nettype wire
