using Common.Factory;
using Common.Models.Responses;
using Microsoft.Extensions.Logging;
using Moq;
using SchedulerService.Service;

namespace SchedulerService.Tests.Services
{
    public class InvokeDaprServiceTests
    {
        private static bool ContainsString(object? v, string expected)
        {
            return v != null && v.ToString() != null && v.ToString()!.Contains(expected);
        }

        [Fact]
        public async Task TriggerExportData_PublishesEventAndLogsInformation()
        {
            // Arrange
            var loggerMock = new Mock<ILogger<InvokeDaprService>>();
            var daprClientMock = new Mock<IDaprClientWrapper>();
            var daprFactoryMock = new Mock<IDaprClientFactory>();
            daprFactoryMock.Setup(f => f.CreateClient()).Returns(daprClientMock.Object);
            daprClientMock.Setup(x => x.PublishEventAsync(
                It.IsAny<string>(),
                It.IsAny<string>(),
                It.IsAny<object>(),
                It.IsAny<CancellationToken>())).Returns(Task.CompletedTask);
            var service = new InvokeDaprService(loggerMock.Object, daprFactoryMock.Object);
            var traceParent = "trace-123";

            // Act
            await service.TriggerExportData(traceParent, CancellationToken.None);

            // Assert
            daprClientMock.Verify(
                x => x.PublishEventAsync(
                    It.IsAny<string>(),
                    It.IsAny<string>(),
                    It.IsAny<object>(),
                    It.IsAny<CancellationToken>()),
                Times.Once);
            loggerMock.Verify(
                x => x.Log(
                    LogLevel.Information,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => ContainsString(v, traceParent)),
                    null,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }

        [Fact]
        public async Task TriggerTrainingOfAiModel_LogsSuccessOnSuccessResponse()
        {
            // Arrange
            var loggerMock = new Mock<ILogger<InvokeDaprService>>();
            var daprClientMock = new Mock<IDaprClientWrapper>();
            var daprFactoryMock = new Mock<IDaprClientFactory>();
            daprFactoryMock.Setup(f => f.CreateClient()).Returns(daprClientMock.Object);
            daprClientMock.Setup(x => x.InvokeMethodAsync<TrainAiModelResponse>(
                It.IsAny<HttpMethod>(),
                It.IsAny<string>(),
                It.IsAny<string>(),
                It.IsAny<CancellationToken>()))
                .ReturnsAsync(new TrainAiModelResponse { Success = true });
            var service = new InvokeDaprService(loggerMock.Object, daprFactoryMock.Object);

            // Act
            await service.TriggerTrainingOfAiModel(CancellationToken.None);

            // Assert
            daprClientMock.Verify(x => x.InvokeMethodAsync<TrainAiModelResponse>(
                It.IsAny<HttpMethod>(),
                It.IsAny<string>(),
                It.IsAny<string>(),
                It.IsAny<CancellationToken>()), Times.Once);
            loggerMock.Verify(
                x => x.Log(
                    LogLevel.Information,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => ContainsString(v, "Successfully triggered training of AI Model")),
                    null,
                    It.IsAny<Func<It.IsAnyType, Exception?, string>>()),
                Times.Once);
        }
    }
}
