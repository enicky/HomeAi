

using SchedulerService.Service;

namespace SchedulerService.Triggers;

public class TriggerRetrieveDataForAi
{
    public readonly ILogger<TriggerRetrieveDataForAi> _logger;
    private readonly IInvokeDaprService _invokeDaprService;

    public TriggerRetrieveDataForAi(ILogger<TriggerRetrieveDataForAi> logger, IInvokeDaprService invokeDaprService)
    {
        _logger = logger;
        _invokeDaprService = invokeDaprService;
    }

    public async Task RunAsync(CancellationToken token)
    {
        const string logPrefix = "[TriggerRetrieveDataForAi:RunAsync]";
        _logger.LogInformation($"{logPrefix} Process data");
        await _invokeDaprService.TriggerExportData(token);
        _logger.LogInformation($"{logPrefix} Finished processing data remotely");
    }
}