using SchedulerService.Service;

namespace SchedulerService.Triggers;

public class TriggerTrainAiModel{
    private readonly ILogger<TriggerTrainAiModel> _logger;
    private readonly IInvokeDaprService _invokeDaprService;

    public TriggerTrainAiModel(ILogger<TriggerTrainAiModel> logger,  IInvokeDaprService invokeDaprService){
        _logger = logger;
        _invokeDaprService = invokeDaprService;
    }

     public async Task RunAsync(CancellationToken token)
    {
        const string logPrefix = "[TriggerTrainAiModel:RunAsync]";

        _logger.LogInformation($"{logPrefix} Start triggering training of AI model" );
        await _invokeDaprService.TriggerTrainingOfAiModel();
        _logger.LogInformation($"{logPrefix} Finished trigger");
    }


}