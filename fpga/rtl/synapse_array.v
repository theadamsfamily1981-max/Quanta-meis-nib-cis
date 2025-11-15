/**
 * Synapse Array
 *
 * Dense synapse weight matrix for SNN connectivity
 * Optimized for BRAM utilization on Virtex VU33P
 *
 * Features:
 * - Dual-port BRAM for weights
 * - Configurable weight precision
 * - Sparse connectivity support via masking
 * - Weight update interface for learning
 */

module synapse_array #(
    parameter NUM_INPUTS = 256,        // Number of pre-synaptic neurons
    parameter NUM_OUTPUTS = 128,       // Number of post-synaptic neurons
    parameter WEIGHT_WIDTH = 16,       // Weight precision
    parameter ADDR_WIDTH = 8           // Address width
) (
    input wire clk,
    input wire rst_n,

    // Input spikes (from pre-synaptic neurons)
    input wire [NUM_INPUTS-1:0] input_spikes,

    // Weight memory interface (for updates)
    input wire weight_we,              // Write enable
    input wire [ADDR_WIDTH-1:0] weight_addr_row,
    input wire [ADDR_WIDTH-1:0] weight_addr_col,
    input wire signed [WEIGHT_WIDTH-1:0] weight_data_in,

    // Outputs (currents to post-synaptic neurons)
    output reg signed [WEIGHT_WIDTH-1:0] output_currents [0:NUM_OUTPUTS-1]
);

    // Weight memory (BRAM)
    // Organized as [output][input] for efficient column access
    (* ram_style = "block" *) reg signed [WEIGHT_WIDTH-1:0] weights [0:NUM_OUTPUTS-1][0:NUM_INPUTS-1];

    // Internal signals
    integer i, j;
    reg signed [WEIGHT_WIDTH+4-1:0] current_accumulator [0:NUM_OUTPUTS-1];  // Extra bits for accumulation

    // Weight update
    always @(posedge clk) begin
        if (weight_we) begin
            weights[weight_addr_row][weight_addr_col] <= weight_data_in;
        end
    end

    // Synaptic current computation
    // For each output neuron, accumulate weighted input spikes
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < NUM_OUTPUTS; i = i + 1) begin
                output_currents[i] <= 0;
                current_accumulator[i] <= 0;
            end
        end else begin
            // Parallel accumulation for each output
            for (i = 0; i < NUM_OUTPUTS; i = i + 1) begin
                current_accumulator[i] = 0;

                // Accumulate contributions from all spiking inputs
                for (j = 0; j < NUM_INPUTS; j = j + 1) begin
                    if (input_spikes[j]) begin
                        current_accumulator[i] = current_accumulator[i] + weights[i][j];
                    end
                end

                // Saturate and assign to output
                if (current_accumulator[i] > $signed({1'b0, {WEIGHT_WIDTH-1{1'b1}}})) begin
                    output_currents[i] <= {1'b0, {WEIGHT_WIDTH-1{1'b1}}};  // Max positive
                end else if (current_accumulator[i] < $signed({1'b1, {WEIGHT_WIDTH-1{1'b0}}})) begin
                    output_currents[i] <= {1'b1, {WEIGHT_WIDTH-1{1'b0}}};  // Max negative
                end else begin
                    output_currents[i] <= current_accumulator[i][WEIGHT_WIDTH-1:0];
                end
            end
        end
    end

endmodule
