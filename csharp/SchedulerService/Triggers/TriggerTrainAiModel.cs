using Common.Helpers;
using Dapr.Client;
using SchedulerService.Service;

namespace SchedulerService.Triggers;

public class TriggerTrainAiModel{
    private readonly ILogger<TriggerTrainAiModel> _logger;
    private readonly DaprClient _daprClient;    

    public TriggerTrainAiModel(ILogger<TriggerTrainAiModel> logger,  DaprClient daprClient){
        _logger = logger;
        _daprClient = daprClient;
    }

     public async Task RunAsync(CancellationToken token)
    {
        const string logPrefix = "[TriggerTrainAiModel:RunAsync]";

        _logger.LogInformation($"{logPrefix} Start triggering training of AI model" );
        await _daprClient.PublishEventAsync(NameConsts.AI_PUBSUB_NAME, NameConsts.AI_START_TRAIN_MODEL);
        _logger.LogInformation($"{logPrefix} Finished trigger");
    }


}