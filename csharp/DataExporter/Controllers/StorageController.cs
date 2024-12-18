using Common.Helpers;
using Common.Models.AI;
using Common.Services;
using Dapr;
using Dapr.Client;
using DataExporter.Services;
using Microsoft.AspNetCore.Mvc;

namespace DataExporter.Controllers;

[ApiController]
[Route("api/[controller]")]
public class StorageController(ILogger<StorageController> logger, IFileService fileService, ILocalFileService localFileService)
    : ControllerBase
{
    private static string TargetFolder =  $"{Path.DirectorySeparatorChar}app{Path.DirectorySeparatorChar}checkpoints{Path.DirectorySeparatorChar}";

    [Topic(pubsubName: NameConsts.AI_PUBSUB_NAME, name: NameConsts.AI_START_UPLOAD_MODEL)]
    [HttpPost(NameConsts.AI_START_UPLOAD_MODEL)]
    public async Task StartUploadingModelToAzure(
        [FromBody] StartUploadModel startUploadModel,
        [FromServices] DaprClient daprClient,
        CancellationToken token
    )
    {
        if (ModelState.IsValid)
        {
            logger.LogInformation(
                "Trigger received to upload model {ModelPath} to azure",
                startUploadModel.ModelPath
            );
        }
        logger.LogInformation(
            "ModelPath : {ModelPath}, TriggerMoment : {DateTime}",
            startUploadModel.ModelPath,
            startUploadModel.TriggerMoment
        );
        try
        {
            string fileName = Path.GetFileName(startUploadModel.ModelPath);
            logger.LogInformation(
                "FileName retrieved from ... {ModelPath} {FileName}",
                startUploadModel.ModelPath,
                fileName
            );
            var generatedFileName = $"{DateTime.Now.ToString("yyyyMMdd")}-{fileName}";
            var targetFolder = Path.GetDirectoryName(startUploadModel.ModelPath)!;
            var fullPath = Path.Join(targetFolder, generatedFileName);
            logger.LogInformation("Renaming file to {FullPath}", fullPath);

            string fPath = Path.GetFullPath(startUploadModel.ModelPath);
            logger.LogInformation($"fPath : {fPath}, TargetFolder : {TargetFolder}, targetFolder : {targetFolder}");
            if (fPath.StartsWith(TargetFolder) && fullPath.StartsWith(targetFolder))
            {
                await localFileService.CopyFile(startUploadModel.ModelPath, fullPath);
                logger.LogInformation("Start uploading file");
                await fileService.UploadToAzure(StorageHelpers.ModelContainerName, fullPath, token);
                logger.LogInformation("Finished uploading file to Azure");
            }
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "There was an error processing the model to azure");
        }
    }
}
