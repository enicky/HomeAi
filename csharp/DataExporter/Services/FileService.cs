using Azure.Storage.Blobs;
using Azure.Storage;
using Azure;
using Common.Helpers;
using Common.Exceptions;
using DataExporter.Services.Factory;
using Microsoft.Extensions.Logging;
using Azure.Storage.Blobs.Models;

namespace app.Services;

public interface IFileService
{
    Task UploadToAzure(string containerName, string generatedFileName, CancellationToken token = default);
}

public class FileService : IFileService
{
    private readonly BlobServiceClient _blobServiceClient;
    private readonly ILogger<FileService> _logger;

    public FileService(IConfiguration configuration, IBlobServiceClientFactory blobServiceClientFactory, ILogger<FileService> logger)
    {
        _logger = logger;
        var _accountName = configuration.GetValue<string>("accountName");
        if (string.IsNullOrEmpty(_accountName))
        {
            throw new NullReferenceException("FileStorage:accountName cannot be NULL");
        }
        var _accountKey = configuration.GetValue<string>("accountKey");
        if (string.IsNullOrEmpty(_accountKey)) throw new NullReferenceException("FileStorage:accountKey cannot be NULL");

        _blobServiceClient = blobServiceClientFactory.Create(_accountName, _accountKey);

        //_blobServiceClient = GetBlobServiceClient(_accountName, _accountKey);
    }



    private async Task<BlobContainerClient?> EnsureContainer(string containerName, CancellationToken token)
    {
        _logger.LogInformation("[EnsureContainer] get blobContainerClient");
        var containerClient = _blobServiceClient.GetBlobContainerClient(containerName);

        try
        {
            // Create the container
            _logger.LogInformation("[EnsureContainer] Start CreateIfNotExistsAsync");
            await containerClient.CreateIfNotExistsAsync(PublicAccessType.None, null, null, token);
            _logger.LogInformation("[EnsureContainer] Check if existsasync exists");
            var containerExists = await containerClient.ExistsAsync(token);
            if (containerExists.Value)
            {
                _logger.LogInformation("[EnsureContainer] Created container {0}", containerClient.Name);
                return containerClient;
            }
        }
        catch (RequestFailedException e)
        {
            _logger.LogError("[EnsureContainer] HTTP error code {0}: {1}", e.Status, e.ErrorCode);
            _logger.LogError($"[EnsureContainer] {e.Message}");
        }
        _logger.LogInformation("[EnsureContainer] Pretty weird I end up here ");
        return containerClient;
    }
    private async Task UploadFromFileAsync(
                                BlobContainerClient containerClient,
                                string localFilePath,
                                CancellationToken token)
    {

        string fileName = Path.GetFileName(localFilePath);
        _logger.LogInformation($"[UploadFromFileASync] Uploading file {fileName}");
        BlobClient blobClient = containerClient.GetBlobClient(fileName);
        _logger.LogInformation($"[UploadFromFileASync] Start uploading async");
        var uploadResult = await blobClient.UploadAsync(localFilePath, true, token);
        _logger.LogInformation($"[UploadFromFileASync] Finished uploading ... result : {uploadResult}");
    }

    public async Task UploadToAzure(string containerName, string generatedFileName, CancellationToken token)
    {
        _logger.LogInformation($"[UploadToAzure] Start uploading to azure using {containerName} and file {generatedFileName}");
        var result = await EnsureContainer(StorageHelpers.ContainerName, token) ?? throw new EnsureContainerException("result is null");
        _logger.LogInformation($"[UploadToAzure] Ensuring of container finished : {result}");
        await UploadFromFileAsync(result, generatedFileName, token);
        _logger.LogInformation("[UploadToAzure] Finished upload to Azure");
    }
}