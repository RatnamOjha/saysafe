from examples.safety_layer import run_demo


def test_offline_safety_flow_approves_exact_action_then_routes_result_privately():
    confirmation, spoken, private = run_demo()

    assert "Maya" in confirmation and "twenty-five dollars" in confirmation
    assert "confirm" in confirmation
    assert private is not None and "$25" in private and "$480" in private
    assert "$25" not in spoken and "$480" not in spoken
