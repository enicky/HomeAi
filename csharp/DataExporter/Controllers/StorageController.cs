using Common.Helpers;
using Common.Models.AI;
using Common.Services;
using Dapr;
using Dapr.Client;
using Microsoft.AspNetCore.Mvc;

namespace DataExporter.Controllers;

public class StorageController(ILogger<StorageController> logger,
        IFileService fileService) : ControllerBase
{
    [Topic(pubsubName: NameConsts.AI_PUBSUB_NAME, name: NameConsts.AI_START_UPLOAD_MODEL)]
    [HttpPost(NameConsts.AI_START_UPLOAD_MODEL)]
    public async Task StartUploadingModelToAzure(StartUploadModel startUploadModel, [FromServices] DaprClient daprClient, CancellationToken token)
    {
        logger.LogInformation($"[StorageController:StartUploadingModelToAzure] Start uploading model {startUploadModel.ModelPath} to azure");
        // first Rename file to current datetime = model
        string fileName = Path.GetFileName(startUploadModel.ModelPath);
        var generatedFileName = $"{DateTime.Now.ToString("YYYYMMdd")}-{fileName}";
        logger.LogInformation($"[StorageController:StartUploadingModelToAzure] Renaming file to {generatedFileName}");
        System.IO.File.Move(startUploadModel.ModelPath, generatedFileName);
        logger.LogInformation("[StorageController:StartUploadingModelToAzure] Start uploading file");
        await fileService.UploadToAzure(StorageHelpers.ModelContainerName, generatedFileName, token);
        logger.LogInformation("[StorageController:StartUploadingModelToAzure] Finished uploading file to Azure");
    }
}