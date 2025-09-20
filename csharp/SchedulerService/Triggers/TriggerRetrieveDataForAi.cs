using System.Diagnostics;
using SchedulerService.Service;

namespace SchedulerService.Triggers;

public class TriggerRetrieveDataForAi(ILogger<TriggerRetrieveDataForAi> logger, IInvokeDaprService? invokeDaprService)
{
    public async Task RunAsync(CancellationToken token)
    {
        const string logPrefix = "[TriggerRetrieveDataForAi:RunAsync]";
        logger.LogInformation($"{logPrefix} Process data");
        using var activity = new Activity("TriggerRetrieveDataForAi").Start();
        var traceParent = Activity.Current?.Id ?? string.Empty;
        logger.LogInformation("{LogPrefix} Current activity id: '{ActivityId}'", logPrefix, activity.Id);
        if(invokeDaprService is not null)
            await invokeDaprService?.TriggerExportData(traceParent, token)!;
        logger.LogInformation($"{logPrefix} Finished processing data remotely");
    }
}