# Language-instruction templates for the color-order cube-stacking tasks.
#
# Each template is a Python format string with three named slots describing the
# stacking order from bottom to top:
#   {c0} = bottom block color   {c1} = middle block color   {c2} = top block color
#
# A trajectory stores a unified template reference with its concrete color order as
# parameters. The final instruction is rendered at training time by picking a template
# (optionally at random for language augmentation) and filling these slots. This keeps
# the per-trajectory .hdf5 minimal while allowing templates to evolve independently.

# Slot semantics, also written into the dumped JSON so consumers are unambiguous.
SLOT_MEANING = "c0 = bottom block, c1 = middle block, c2 = top block (stacking order)"

# Keep templates phrased so that reading them left-to-right never contradicts the
# bottom->top semantics. Mix several framings for linguistic diversity:
#   - explicit ordered list      - bottom-up narration
#   - top-down narration         - "on top of" chains
#   - imperative / descriptive    - varied verbs (stack/place/put/pile/arrange)
STACK_INSTRUCTION_TEMPLATES = [
    # --- explicit ordered list (bottom -> top) ---
    "Stack the cubes in this order from bottom to top: {c0}, {c1}, {c2}.",
    "Stack the blocks bottom to top as {c0}, {c1}, {c2}.",
    "Build a tower with {c0} at the bottom, {c1} in the middle, and {c2} on top.",
    "Arrange the three cubes in the order {c0}, {c1}, {c2}, bottom to top.",
    "Order the stack from the ground up: {c0}, then {c1}, then {c2}.",
    "From bottom to top the cubes should read {c0}, {c1}, {c2}.",
    # --- bottom-up narration ---
    "Place the {c0} block first, then stack the {c1} block on it, and finally the {c2} block on top.",
    "Start with the {c0} cube, add the {c1} cube above it, then top it with the {c2} cube.",
    "Put down the {c0} block, set the {c1} block on it, and finish with the {c2} block.",
    "Lay the {c0} cube as the base, stack {c1} over it, and place {c2} at the very top.",
    "First the {c0} block goes down, next the {c1} block, and last the {c2} block.",
    "Begin by placing {c0}, follow with {c1}, and end by stacking {c2}.",
    # --- top-down narration ---
    "Put the {c2} block on top of the {c1} block, which rests on the {c0} block.",
    "The {c2} cube goes on the {c1} cube, and the {c1} cube goes on the {c0} cube.",
    "Place {c2} above {c1}, and {c1} above {c0}.",
    "Stack {c2} onto {c1}, with {c1} already sitting on {c0}.",
    # --- "on top of" chains ---
    "Stack {c1} on {c0}, then {c2} on {c1}.",
    "Put the {c1} cube on the {c0} cube and the {c2} cube on the {c1} cube.",
    "Set {c1} on top of {c0}, and {c2} on top of {c1}.",
    # --- compact imperative ---
    "Make a stack: {c0} bottom, {c1} middle, {c2} top.",
    "Pile the cubes {c0} - {c1} - {c2} from bottom up.",
    "Stack order (low to high): {c0}, {c1}, {c2}.",
    # --- descriptive / goal-state ---
    "The finished tower has {c0} on the bottom, {c1} in the center, and {c2} crowning it.",
    "When done, {c0} supports {c1}, and {c1} supports {c2}.",
    "Assemble a three-block tower whose layers are {c0}, {c1}, {c2} going upward.",
    "Transfer and stack the blocks so the order is {c0}, {c1}, {c2} from bottom to top.",
    # --- task-flavored (matches the bi-manual transfer-and-stack scenario) ---
    "Hand over and stack the cubes in the order {c0}, {c1}, {c2} (bottom to top).",
    "Pass each block to the stack so they pile up as {c0}, {c1}, {c2}.",
    "Stack the colored cubes from bottom to top: {c0} first, {c2} last.",
    "Build the tower in sequence {c0} -> {c1} -> {c2}, lowest to highest.",
]


def render(template, color_order):
    """Fill a template with a concrete bottom->top color order.

    Args:
        template: a format string using {c0}, {c1}, {c2}.
        color_order: sequence of 3 color names, bottom->top, e.g. ["blue","green","red"].
    Returns:
        The rendered instruction string.
    """
    c0, c1, c2 = color_order
    return template.format(c0=c0, c1=c1, c2=c2)


def build_instruction_library():
    """Return the JSON-serializable dict written to dataset_dir/instruction_library.json.

    This creates a unified instruction library with one template task for the color-order
    cube-stacking scenario.
    """
    return {
        "0": {
            "name": "stack_cubes",
            "type": "template",
            "slot_meaning": SLOT_MEANING,
            "templates": list(STACK_INSTRUCTION_TEMPLATES),
        }
    }
