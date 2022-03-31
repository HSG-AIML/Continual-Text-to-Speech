from .naive import Naive


def get_strategy(
        params,
        config,
        model,
        optimizer,
        criterion,
        plugins,
        evaluation_plugin,
        device
):

    if params["strategy"] == "naive":
        strategy = Naive(
            model, optimizer, criterion,
            train_mb_size=config.batch_size,
            train_epochs=config.epochs,
            device=device,
            plugins=plugins,
            evaluator=evaluation_plugin,
        )
    else:
        raise NotImplementedError

    return strategy
