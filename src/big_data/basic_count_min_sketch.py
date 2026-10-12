from big_data.sketches.count_min import CountMinSketch


# Demo 1: Basic CMS functionality
def demo_basic_cms():
    """Demonstrate basic CMS functionality for switching counting"""
    print("=" * 60)
    print("DEMO 1: Basic Count-Min Sketch for Switching Activity")
    print("=" * 60)

    # Create CMS with small dimensions for demo
    cms = CountMinSketch(width=50, depth=3)

    # Simulate toggles for different nodes
    nodes = {
        "clk_buffer_1": 1001,
        "clk_buffer_2": 1002,
        "data_reg_1": 2001,
        "data_reg_2": 2002,
        "alu_out": 3001,
        "memory_cell": 4001,
    }

    # Simulate toggle events
    print("\nSimulating toggle events...")
    toggle_events = []

    # Node 1001 toggles frequently (high activity)
    for _ in range(150):
        cms.update(nodes["clk_buffer_1"])
        toggle_events.append(("clk_buffer_1", nodes["clk_buffer_1"]))

    # Node 2001 toggles moderately
    for _ in range(75):
        cms.update(nodes["data_reg_1"])
        toggle_events.append(("data_reg_1", nodes["data_reg_1"]))

    # Node 4001 toggles rarely (low activity)
    for _ in range(10):
        cms.update(nodes["memory_cell"])
        toggle_events.append(("memory_cell", nodes["memory_cell"]))

    print(f"Total toggle events simulated: {len(toggle_events)}")

    # Query estimates
    print("\nEstimated toggle counts:")
    for name, node_id in nodes.items():
        estimate = cms.query(node_id)
        print(f"  {name:15s} (ID: {node_id}): {estimate:.1f}")

    # Show sketch visualization
    print("\nSketch visualization (first 10 counters of each row):")
    for d in range(cms.depth):
        row_preview = cms.sketch[d, :10]
        print(f"  Row {d}: {row_preview}")

    return cms, nodes


if __name__ == "__main__":
    # Run demo 1
    cms_basic, node_ids = demo_basic_cms()
