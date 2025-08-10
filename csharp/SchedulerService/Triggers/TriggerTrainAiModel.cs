using Common.Helpers;
using Dapr.Client;
using SchedulerService.Service;

namespace SchedulerService.Triggers;

public class TriggerTrainAiModel(ILogger<TriggerTrainAiModel> logger, DaprClient daprClient)
{
    public async Task RunAsync(CancellationToken token)
    {
        const string logPrefix = "[TriggerTrainAiModel:RunAsync]";

        logger.LogInformation($"{logPrefix} Start triggering training of AI model" );
        await daprClient.PublishEventAsync(NameConsts.AI_PUBSUB_NAME, NameConsts.AI_START_TRAIN_MODEL, token);
        logger.LogInformation($"{logPrefix} Finished trigger");
    }


}