using System.Diagnostics;
using Common.Helpers;
using Common.Models.AI;
using Common.Models.Responses;
using Dapr;
using Dapr.Client;
using Microsoft.AspNetCore.Mvc;
using Newtonsoft.Json;

namespace SchedulerService.Controllers;

[Route("api/[controller]")]
[ApiController]
public class DaprController : ControllerBase
{
    private readonly ILogger<DaprController> logger;
    private readonly DaprClient _daprClient;

    public DaprController(ILogger<DaprController> logger, DaprClient daprClient)
    {
        this.logger = logger;
        _daprClient = daprClient;
    }

    [Topic(NameConsts.INFLUX_PUBSUB_NAME, "testreply")]
    [HttpPost("testreply")]
    public IActionResult TestReply([FromBody] RetrieveDataResponse? o)
    {
        logger.LogInformation($"TestReply got triggered");
        if (o is not null)
        {
            var success= o.Success.ToString();
            var generatedFileName = o.GeneratedFileName.Replace(" ", "_");
            var startAiProcess = o.StartAiProcess.ToString();
            logger.LogInformation("Received RetrieveDataResponse response : {Success} for id {GeneratedFileName} -> {StartAiProcess}", success, generatedFileName, startAiProcess);
            return Ok();
        }
        return BadRequest();
    }


    [Topic(pubsubName: NameConsts.INFLUX_PUBSUB_NAME, name: NameConsts.INFLUX_FINISHED_RETRIEVE_DATA)]
    [HttpPost("DownloadDataHasFinished")]
    public async Task DownloadDataHasFinished([FromBody] RetrieveDataResponse? response)
    {
        logger.LogInformation("Trigger received that download has Finished {ResponseValue}", JsonConvert.SerializeObject(response));
        logger.LogInformation("Completed {IsCompleted}, filename {FileName}",response?.Success, response?.GeneratedFileName);
        logger.LogInformation("Start AI Processing ? {StartAiProcessing}", response?.StartAiProcess);
        logger.LogInformation("TraceParent {TraceParent}", response?.TraceParent);
        var mustTrainModel = false;
        if(response == null)
        {
            logger.LogWarning("Response was null");
            return;
        }
        if (!response.Success)
        {
            logger.LogWarning("Download of data pre training AI model failed !!");
            return;
        }
        if (response.StartAiProcess)
        {
            logger.LogInformation("Start training model was true => set boolean val");
            mustTrainModel = true;
        }

        if (mustTrainModel)
        {
            logger.LogInformation("Start triggering of downloading data to python training container");

            var evt = new StartDownloadDataEvent { TraceParent = response.TraceParent, TraceState = response.TraceState};
            var evtJson = JsonConvert.SerializeObject(evt);
            logger.LogInformation("StartDownloadDataEvent: {EventJson}", evtJson);
            await _daprClient.PublishEventAsync(NameConsts.AI_PUBSUB_NAME, NameConsts.AI_START_DOWNLOAD_DATA, evtJson);
            logger.LogInformation("Sent message to AI container to start Downloading data and prepare it to start training model");
        }
        else
        {
            logger.LogInformation("Training model was not needed so skipped");
        }
        logger.LogInformation("Finished exchanging messages");
    }

    [Topic(NameConsts.AI_PUBSUB_NAME, NameConsts.AI_FINISHED_DOWNLOAD_DATA)]
    [HttpPost("AiDownloadFinishedStartTraining")]
    public async Task AiDownloadFinishedStartTraining()
    {
        logger.LogInformation("Retrieved from python module that download of data has been finished");
        logger.LogInformation("Send message to start training model on AI container");
        var traceParent = Activity.Current?.Id ?? string.Empty;
        var evt = new StartTrainModelEvent { TraceParent = traceParent, TraceState = Activity.Current?.TraceStateString ?? string.Empty};
        await _daprClient.PublishEventAsync(NameConsts.AI_PUBSUB_NAME, NameConsts.AI_START_TRAIN_MODEL, evt);
        logger.LogInformation("Finished sending message to AI container to start training model");

    }

    [Topic(NameConsts.AI_PUBSUB_NAME, NameConsts.AI_FINISHED_TRAIN_MODEL)]
    [HttpPost("AiFinishedTrainingModel")]
    public async Task AiFinishedTrainingModel([FromBody] TrainAiModelResponse response)
    {
        var traceParent = Activity.Current?.Id ?? string.Empty;
        logger.LogInformation("Retrieved message that training of model has been finished");
        logger.LogInformation("Training was a success: {IsSuccess}", response.Success);
        logger.LogInformation("ModelPath to use is : {ModelPath}", response.ModelPath);
        
        if (response != null && !string.IsNullOrEmpty(response.ModelPath))
        {
            logger.LogInformation("Can start uploading model to Azure");
            var data = new StartUploadModel
            {
                ModelPath = response.ModelPath,
                TriggerMoment = DateTime.Now,
                TraceParent = traceParent, 
                TraceState = Activity.Current?.TraceStateString ?? string.Empty
            };
            await _daprClient.PublishEventAsync(NameConsts.AI_PUBSUB_NAME, NameConsts.AI_START_UPLOAD_MODEL, data);
            logger.LogInformation("Finished sending message to upload model to azure");

        }

        logger.LogInformation("Process finished");
    }
}
