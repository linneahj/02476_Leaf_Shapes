from timm.models import create_model as timm_create_model


def create_model(
    model_name: str,
    **model_kwargs,
):
    model = timm_create_model(model_name, model_kwargs)
    return model
