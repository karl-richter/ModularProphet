from modularprophet.components import Component


def validate_inputs(items, instances):
    if not isinstance(instances, list):
        instances = [instances]
    for item in items:
        if not any([isinstance(item, instance) for instance in instances]):
            raise TypeError(
                f"The type {type(item).__module__} is not allowed. Use any of {', '.join([instance.__module__ for instance in instances])}."
            )


def models_to_summary(name, items):
    if not isinstance(items, list):
        items = [items]
    return f"{name}(\n" + "\n".join(["\t" + c.__repr__() for c in items]) + "\n)"


def components_to_summary(name, items):
    return (
        f"{name}(\n"
        + "\n".join(
            [
                "\t\t(" + (c.id if c.id else c.name.lower()) + "): " + c.__repr__()
                for c in items
            ]
        )
        + "\n\t)"
    )
