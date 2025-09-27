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

public class InvokeDaprService(ILogger<InvokeDaprService> logger, IDaprClientFactory daprClientFactory)
    : IInvokeDaprService
{
    public async Task TriggerExportData(string traceParent, CancellationToken token = default)
    {
        var client = daprClientFactory.CreateClient();
        var evt = new StartDownloadDataEvent { TraceParent = traceParent };
        await client.PublishEventAsync(NameConsts.INFLUX_PUBSUB_NAME, NameConsts.INFLUX_RETRIEVE_DATA, evt, token);
        logger.LogInformation("Send event to retrieve data with traceParent: {TraceParent}", traceParent);
    }

    public async Task TriggerTrainingOfAiModel(CancellationToken token = default)
    {
        var client = daprClientFactory.CreateClient();
        var retrieveResponse = await client.InvokeMethodAsync<TrainAiModelResponse>(HttpMethod.Get, "pythonaitrainer", "/train_model", token);
        if (retrieveResponse.Success)
        {
            logger.LogInformation("Successfully triggered training of AI Model");
            return;
        }
        logger.LogWarning("There was an issue triggering the training of the AI model");
    }
}