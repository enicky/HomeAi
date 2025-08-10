using Common.Factory;

namespace SchedulerService.Service;

using Common.Helpers;
using Common.Models.AI;
using Common.Models.Responses;

public interface IInvokeDaprService
{
    Task TriggerExportData(string traceParent, CancellationToken token = default);
    Task TriggerTrainingOfAiModel(CancellationToken token = default);
}

public class InvokeDaprService : IInvokeDaprService
{
    private readonly ILogger<InvokeDaprService> _logger;
    private readonly IDaprClientFactory _daprClientFactory;

    public InvokeDaprService(ILogger<InvokeDaprService> logger, IDaprClientFactory daprClientFactory)
    {
        _logger = logger;
        _daprClientFactory = daprClientFactory;
    }

    public async Task TriggerExportData(string traceParent, CancellationToken token = default)
    {
        var client = _daprClientFactory.CreateClient();
        var evt = new StartDownloadDataEvent { TraceParent = traceParent };
        await client.PublishEventAsync(NameConsts.INFLUX_PUBSUB_NAME, NameConsts.INFLUX_RETRIEVE_DATA, evt, token);
        _logger.LogInformation("Send event to retrieve data with traceParent: {TraceParent}", traceParent);
    }

    public async Task TriggerTrainingOfAiModel(CancellationToken token = default)
    {
        var client = _daprClientFactory.CreateClient();
        var retrieveResponse = await client.InvokeMethodAsync<TrainAiModelResponse>(HttpMethod.Get, "pythonaitrainer", "/train_model", token);
        if (retrieveResponse.Success)
        {
            _logger.LogInformation("Successfully triggered training of AI Model");
            return;
        }
        _logger.LogWarning("There was an issue triggering the training of the AI model");
    }
}