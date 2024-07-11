using Common.Helpers;
using Common.Models;
using Dapr.Client;
using SchedulerService.Service;

namespace SchedulerService.Triggers;

public class TriggerTestDapr{
    private readonly ILogger<TriggerTestDapr> _logger;
    private readonly DaprClient _daprClient;

    public TriggerTestDapr(ILogger<TriggerTestDapr> logger, DaprClient daprClient){
        _logger = logger;
        _daprClient = daprClient;
    }

     public async Task RunAsync(CancellationToken token)
    {
        const string logPrefix = "[TriggerTestDapr:RunAsync]";

        _logger.LogInformation($"{logPrefix} start trigger test" );
        
        await _daprClient.PublishEventAsync(NameConsts.AI_PUBSUB_NAME, "test", new Order{Id = 1, Title="test"});
        
        _logger.LogInformation($"{logPrefix} Finished trigger");
    }


}