import logging
from modularprophet.components import Component

logger = logging.getLogger("experiments")


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


def get_n_lags_from_model(model):
    n_lags = 0
    for component in model.models:
        if hasattr(component, "n_lags"):
            n_lags = component.n_lags
    if n_lags == 0:
        logger.warning(
            "The model does not contain any components with n_lags attribute."
        )
    return n_lags
