/**
 * SNN Layer
 *
 * Complete spiking neural network layer with neurons and synapses
 * Optimized for Virtex UltraScale+ VU33P
 *
 * Features:
 * - Configurable number of neurons
 * - Fully-connected or sparse connectivity
 * - Spike encoding/decoding
 * - Performance counters
 */

module snn_layer #(
    parameter NUM_NEURONS = 128,
    parameter NUM_INPUTS = 256,
    parameter WEIGHT_WIDTH = 16,
    parameter NEURON_WIDTH = 16,
    parameter THRESHOLD = 16'h0100,
    parameter LEAK = 16'h00F0
) (
    input wire clk,
    input wire rst_n,
    input wire enable,

    // Input spikes from previous layer
    input wire [NUM_INPUTS-1:0] input_spikes,

    // Weight programming interface
    input wire weight_we,
    input wire [7:0] weight_addr_row,
    input wire [7:0] weight_addr_col,
    input wire signed [WEIGHT_WIDTH-1:0] weight_data,

    // Output spikes to next layer
    output wire [NUM_NEURONS-1:0] output_spikes,

    // Status outputs
    output wire [NUM_NEURONS-1:0] refractory_status,
    output reg [31:0] spike_count,           // Total spike count
    output reg [31:0] cycle_count            // Time step counter
);

    // Synaptic currents (from synapse array to neurons)
    wire signed [WEIGHT_WIDTH-1:0] synaptic_currents [0:NUM_NEURONS-1];

    // Membrane potentials (for monitoring)
    wire signed [NEURON_WIDTH-1:0] membrane_potentials [0:NUM_NEURONS-1];

    // Synapse array instance
    synapse_array #(
        .NUM_INPUTS(NUM_INPUTS),
        .NUM_OUTPUTS(NUM_NEURONS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH)
    ) synapse_inst (
        .clk(clk),
        .rst_n(rst_n),
        .input_spikes(input_spikes),
        .weight_we(weight_we),
        .weight_addr_row(weight_addr_row),
        .weight_addr_col(weight_addr_col),
        .weight_data_in(weight_data),
        .output_currents(synaptic_currents)
    );

    // Neuron array (parallel instantiation)
    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin : neuron_array
            lif_neuron #(
                .WIDTH(NEURON_WIDTH),
                .THRESHOLD(THRESHOLD),
                .LEAK(LEAK)
            ) neuron_inst (
                .clk(clk),
                .rst_n(rst_n),
                .enable(enable),
                .input_current(synaptic_currents[i]),
                .threshold_override(16'h0),
                .threshold_override_en(1'b0),
                .spike_out(output_spikes[i]),
                .membrane_potential(membrane_potentials[i]),
                .refractory(refractory_status[i])
            );
        end
    endgenerate

    // Performance counters
    integer j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spike_count <= 32'b0;
            cycle_count <= 32'b0;
        end else if (enable) begin
            // Count total spikes this cycle
            spike_count <= spike_count + count_ones(output_spikes);

            // Increment cycle counter
            cycle_count <= cycle_count + 1'b1;
        end
    end

    // Helper function to count set bits
    function integer count_ones;
        input [NUM_NEURONS-1:0] bits;
        integer k;
        begin
            count_ones = 0;
            for (k = 0; k < NUM_NEURONS; k = k + 1) begin
                if (bits[k]) count_ones = count_ones + 1;
            end
        end
    endfunction

endmodule
