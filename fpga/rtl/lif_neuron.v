/**
 * Leaky Integrate-and-Fire (LIF) Neuron
 *
 * Hardware implementation for Virtex UltraScale+ VU33P
 *
 * Features:
 * - Configurable membrane potential dynamics
 * - Threshold-based spiking
 * - Refractory period support
 * - 16-bit fixed-point arithmetic
 */

module lif_neuron #(
    parameter WIDTH = 16,              // Data width (fixed-point)
    parameter FRAC_BITS = 8,           // Fractional bits
    parameter THRESHOLD = 16'h0100,    // Spike threshold (1.0 in fixed-point)
    parameter LEAK = 16'h00F0,         // Leak factor (0.94 in fixed-point)
    parameter RESET_POTENTIAL = 16'h0000,  // Reset potential after spike
    parameter REFRAC_CYCLES = 4        // Refractory period cycles
) (
    input wire clk,
    input wire rst_n,
    input wire enable,

    // Input current (signed fixed-point)
    input wire signed [WIDTH-1:0] input_current,

    // Control signals
    input wire signed [WIDTH-1:0] threshold_override,  // Dynamic threshold
    input wire threshold_override_en,

    // Outputs
    output reg spike_out,              // 1 when neuron spikes
    output reg signed [WIDTH-1:0] membrane_potential,
    output reg refractory            // 1 when in refractory period
);

    // Internal registers
    reg signed [WIDTH-1:0] v_mem;         // Membrane potential
    reg [3:0] refrac_counter;              // Refractory counter
    reg signed [WIDTH-1:0] threshold_reg;  // Actual threshold to use

    // Intermediate signals
    wire signed [WIDTH-1:0] leak_decay;
    wire signed [WIDTH-1:0] v_next;
    wire spike_condition;

    // Leak decay: v_mem * LEAK
    // Use DSP48E2 slice for multiplication (optimized for UltraScale+)
    wire signed [2*WIDTH-1:0] leak_product;
    assign leak_product = v_mem * LEAK;
    assign leak_decay = leak_product[FRAC_BITS +: WIDTH];  // Scale back

    // Next membrane potential: leak_decay + input_current
    assign v_next = leak_decay + input_current;

    // Spike condition: v_mem >= threshold (when not in refractory)
    assign spike_condition = (v_mem >= threshold_reg) && !refractory && enable;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v_mem <= RESET_POTENTIAL;
            spike_out <= 1'b0;
            refractory <= 1'b0;
            refrac_counter <= 4'b0;
            membrane_potential <= RESET_POTENTIAL;
            threshold_reg <= THRESHOLD;
        end else begin
            // Update threshold
            if (threshold_override_en)
                threshold_reg <= threshold_override;
            else
                threshold_reg <= THRESHOLD;

            if (enable) begin
                // Handle refractory period
                if (refractory) begin
                    if (refrac_counter == REFRAC_CYCLES - 1) begin
                        refractory <= 1'b0;
                        refrac_counter <= 4'b0;
                    end else begin
                        refrac_counter <= refrac_counter + 1'b1;
                    end
                    spike_out <= 1'b0;
                    // Membrane potential stays at reset during refractory
                end
                // Spike condition
                else if (spike_condition) begin
                    v_mem <= RESET_POTENTIAL;
                    spike_out <= 1'b1;
                    refractory <= 1'b1;
                    refrac_counter <= 4'b0;
                end
                // Normal integration
                else begin
                    v_mem <= v_next;
                    spike_out <= 1'b0;
                end

                membrane_potential <= v_mem;
            end else begin
                spike_out <= 1'b0;
            end
        end
    end

endmodule
