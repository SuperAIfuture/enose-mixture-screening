`timescale 1ns/1ps
`default_nettype none

// C2 frozen fixed-point common-vector core.
// Weight and vector memory files are generated beside this source.
module tloc_qsnn_fixed_core #(
    parameter int INPUT_FEATURES = 800,
    parameter int HIDDEN_NEURONS = 64,
    parameter int OUTPUT_CLASSES = 7,
    parameter int TIME_STEPS = 30,
    parameter int FRACTIONAL_BITS = 10,
    parameter int HIDDEN_COEFFICIENT_Q = 928,
    parameter int OUTPUT_COEFFICIENT_Q = 1142,
    parameter int MEMBRANE_THRESHOLD_Q = 128
)(
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         start_sample,
    input  logic                         frame_valid,
    input  logic [INPUT_FEATURES-1:0]    frame_in,
    output logic                         busy,
    output logic                         done,
    output logic [$clog2(OUTPUT_CLASSES)-1:0] class_idx,
    output logic [OUTPUT_CLASSES-1:0][15:0] output_counts
);

    localparam int HIDDEN_WEIGHT_COUNT = INPUT_FEATURES * HIDDEN_NEURONS;
    localparam int OUTPUT_WEIGHT_COUNT = HIDDEN_NEURONS * OUTPUT_CLASSES;
    localparam longint signed UPDATE_DENOMINATOR = (2 <<< FRACTIONAL_BITS);

    logic signed [7:0] hidden_weight [0:HIDDEN_WEIGHT_COUNT-1];
    logic signed [7:0] output_weight [0:OUTPUT_WEIGHT_COUNT-1];
    initial begin
        $readmemh("hidden_weights.mem", hidden_weight);
        $readmemh("output_weights.mem", output_weight);
    end

    logic signed [63:0] hidden_membrane_q [0:HIDDEN_NEURONS-1];
    logic signed [63:0] output_membrane_q [0:OUTPUT_CLASSES-1];
    logic [15:0] output_count_q [0:OUTPUT_CLASSES-1];
    logic [$clog2(TIME_STEPS)-1:0] frame_count;

    logic signed [63:0] hidden_dot_comb [0:HIDDEN_NEURONS-1];
    logic signed [63:0] hidden_membrane_next [0:HIDDEN_NEURONS-1];
    logic hidden_spike_comb [0:HIDDEN_NEURONS-1];
    logic signed [63:0] output_dot_comb [0:OUTPUT_CLASSES-1];
    logic signed [63:0] output_membrane_next [0:OUTPUT_CLASSES-1];
    logic output_spike_comb [0:OUTPUT_CLASSES-1];
    logic [15:0] final_count_comb [0:OUTPUT_CLASSES-1];
    logic [$clog2(OUTPUT_CLASSES)-1:0] final_class_comb;

    function automatic logic signed [63:0] fixed_update(
        input logic signed [63:0] membrane_q,
        input logic signed [63:0] dot_product,
        input integer coefficient_q
    );
        logic signed [63:0] numerator;
        logic signed [63:0] absolute_value;
        logic signed [63:0] quotient;
        logic signed [63:0] remainder;
        logic signed [63:0] sign_value;
        logic round_up;
        begin
            numerator = membrane_q * (64'sd1 <<< FRACTIONAL_BITS)
                      + (64'sd2 * dot_product * coefficient_q);
            if (numerator < 0) begin
                sign_value = -64'sd1;
                absolute_value = -numerator;
            end
            else begin
                sign_value = 64'sd1;
                absolute_value = numerator;
            end
            quotient = absolute_value / UPDATE_DENOMINATOR;
            remainder = absolute_value % UPDATE_DENOMINATOR;
            round_up = (remainder > (UPDATE_DENOMINATOR / 2))
                    || ((remainder == (UPDATE_DENOMINATOR / 2)) && quotient[0]);
            fixed_update = sign_value * (quotient + (round_up ? 64'sd1 : 64'sd0));
        end
    endfunction

    always_comb begin
        for (int h = 0; h < HIDDEN_NEURONS; h++) begin
            hidden_dot_comb[h] = 64'sd0;
            for (int j = 0; j < INPUT_FEATURES; j++) begin
                if (frame_in[j])
                    hidden_dot_comb[h] = hidden_dot_comb[h] + $signed(hidden_weight[h*INPUT_FEATURES+j]);
            end
            hidden_membrane_next[h] = fixed_update(
                hidden_membrane_q[h], hidden_dot_comb[h], HIDDEN_COEFFICIENT_Q
            );
            hidden_spike_comb[h] = (hidden_membrane_next[h] > MEMBRANE_THRESHOLD_Q);
        end
        for (int c = 0; c < OUTPUT_CLASSES; c++) begin
            output_dot_comb[c] = 64'sd0;
            for (int h = 0; h < HIDDEN_NEURONS; h++) begin
                if (hidden_spike_comb[h])
                    output_dot_comb[c] = output_dot_comb[c] + $signed(output_weight[c*HIDDEN_NEURONS+h]);
            end
            output_membrane_next[c] = fixed_update(
                output_membrane_q[c], output_dot_comb[c], OUTPUT_COEFFICIENT_Q
            );
            output_spike_comb[c] = (output_membrane_next[c] > MEMBRANE_THRESHOLD_Q);
            final_count_comb[c] = output_count_q[c] + (output_spike_comb[c] ? 16'd1 : 16'd0);
        end
        final_class_comb = '0;
        for (int c = 1; c < OUTPUT_CLASSES; c++) begin
            if (final_count_comb[c] > final_count_comb[final_class_comb])
                final_class_comb = c;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            class_idx <= '0;
            frame_count <= '0;
            for (int h = 0; h < HIDDEN_NEURONS; h++)
                hidden_membrane_q[h] <= 64'sd0;
            for (int c = 0; c < OUTPUT_CLASSES; c++) begin
                output_membrane_q[c] <= 64'sd0;
                output_count_q[c] <= 16'd0;
                output_counts[c] <= 16'd0;
            end
        end
        else begin
            done <= 1'b0;
            if (start_sample && !busy) begin
                busy <= 1'b1;
                frame_count <= '0;
                for (int h = 0; h < HIDDEN_NEURONS; h++)
                    hidden_membrane_q[h] <= 64'sd0;
                for (int c = 0; c < OUTPUT_CLASSES; c++) begin
                    output_membrane_q[c] <= 64'sd0;
                    output_count_q[c] <= 16'd0;
                    output_counts[c] <= 16'd0;
                end
            end
            else if (busy && frame_valid) begin
                for (int h = 0; h < HIDDEN_NEURONS; h++)
                    hidden_membrane_q[h] <= hidden_spike_comb[h] ? 64'sd0 : hidden_membrane_next[h];
                for (int c = 0; c < OUTPUT_CLASSES; c++) begin
                    output_membrane_q[c] <= output_spike_comb[c] ? 64'sd0 : output_membrane_next[c];
                    output_count_q[c] <= final_count_comb[c];
                end
                if (frame_count == TIME_STEPS-1) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    class_idx <= final_class_comb;
                    for (int c = 0; c < OUTPUT_CLASSES; c++)
                        output_counts[c] <= final_count_comb[c];
                end
                else begin
                    frame_count <= frame_count + 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
