using Common.Helpers;
using Common.Models.AI;
using Common.Models.Responses;
using Dapr.Client;

namespace SchedulerService.Service;

public interface IInvokeDaprService
{
    Task TriggerExportData(string traceParent, CancellationToken token = default);
    Task TriggerTrainingOfAiModel(CancellationToken token = default);
}

public class InvokeDaprService : IInvokeDaprService
{
    private readonly ILogger<InvokeDaprService> _logger;
    public InvokeDaprService(ILogger<InvokeDaprService> logger)
    {
        _logger = logger;
    }

    public async Task TriggerExportData(string traceParent, CancellationToken token = default)
    {
        using var client = new DaprClientBuilder().Build();
        var evt = new StartDownloadDataEvent { TraceParent = traceParent };
        await client.PublishEventAsync(NameConsts.INFLUX_PUBSUB_NAME, NameConsts.INFLUX_RETRIEVE_DATA, evt, token);
        _logger.LogInformation("Send event to retrieve data with traceParent: {TraceParent}", traceParent);
    }

    public async Task TriggerTrainingOfAiModel(CancellationToken token = default)
    {
        using var client = new DaprClientBuilder().Build();
        var retrieveResponse = await client.InvokeMethodAsync<TrainAiModelResponse>(HttpMethod.Get, "pythonaitrainer", "/train_model", token);
        if (retrieveResponse.Success)
        {
            _logger.LogInformation("Successfully triggered training of AI Model");
            return;
        }
        _logger.LogWarning("There was an issue triggering the training of the AI model");
    }
}