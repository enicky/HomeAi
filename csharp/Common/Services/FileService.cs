using Azure.Storage.Blobs;
using Azure;
using Microsoft.Extensions.Configuration;
using Common.Exceptions;
using Common.Factory;
using Common.Helpers;
using Azure.Storage.Blobs.Models;
using Microsoft.Extensions.Logging;

namespace Common.Services;

public interface IFileService
{
    Task<string> RetrieveParsedFile(string fileName, string containerName);
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
            throw new AccountNameNullException("FileStorage:accountName cannot be NULL");
        }
        var _accountKey = configuration.GetValue<string>("accountKey");
        if (string.IsNullOrEmpty(_accountKey))
        {
            throw new AccountKeyNullException("FileStorage:accountKey cannot be NULL");
        }
        
        _blobServiceClient = blobServiceClientFactory.Create(_accountName, _accountKey);
    }

    private async Task<BlobContainerClient> EnsureContainer(string containerName, CancellationToken token)
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
                _logger.LogInformation("[EnsureContainer] Created container {Name}", containerClient.Name);
                return containerClient;
            }
        }
        catch (RequestFailedException e)
        {
            _logger.LogError(e, "[EnsureContainer] HTTP error code {Status}: {ErrorCode}, {Message}", e.Status, e.ErrorCode, e.Message);

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
        _logger.LogInformation("[UploadFromFileASync] Uploading file {FileName}", fileName);
        BlobClient blobClient = containerClient.GetBlobClient(fileName);
        var uploadResult = await blobClient.UploadAsync(localFilePath, true, token);
        _logger.LogInformation("[UploadFromFileASync] Finished uploading ... result : {UploadResult}", uploadResult);
    }

    public async Task UploadToAzure(string containerName, string generatedFileName, CancellationToken token = default)
    {
        _logger.LogInformation("[UploadToAzure] Start uploading to azure using {ContainerName} and file {GeneratedFileName}", containerName, generatedFileName);
        var result = await EnsureContainer(containerName, token);
        await UploadFromFileAsync(result, generatedFileName, token);
        _logger.LogInformation("[UploadToAzure] Finished upload to Azure");
    }

    public async Task<string> RetrieveParsedFile(string fileName, string containerName)
    {
        var containerClient = _blobServiceClient.GetBlobContainerClient(containerName);
        var blobClient = containerClient.GetBlobClient(fileName);
        await blobClient.DownloadToAsync(fileName);      
        return fileName;
        
    }
}