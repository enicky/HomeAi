using System.Diagnostics;
using SchedulerService.Service;

namespace SchedulerService.Triggers;

public class TriggerRetrieveDataForAi(ILogger<TriggerRetrieveDataForAi> logger, IInvokeDaprService? invokeDaprService)
{
    const string LOG_PREFIX = "[TriggerRetrieveDataForAi:RunAsync]";
    public async Task RunAsync(CancellationToken token)
    {
        using var activity = new Activity("TriggerRetrieveDataForAi").Start();
        if(Activity.Current?.Id != null)
            activity.SetParentId(Activity.Current.Id);
        logger.LogInformation("{LogPrefix} Process data", LOG_PREFIX);
        
        logger.LogInformation("{LogPrefix} Current activity id: '{ActivityId}'", LOG_PREFIX, activity.Id);
        if(invokeDaprService is not null){
            logger.LogInformation("{LOG_PREFIX} Calling InfluxDB to retrieve data", LOG_PREFIX);
            await invokeDaprService?.TriggerExportData(Activity.Current?.Id ?? "EMPTY", token)!;
        }
        logger.LogInformation("{LOG_PREFIX} Finished processing data remotely", LOG_PREFIX);
    }
}