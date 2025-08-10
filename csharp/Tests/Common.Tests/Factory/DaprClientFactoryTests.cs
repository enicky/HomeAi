using Common.Factory;
using Dapr.Client;
using Moq;

namespace Common.Tests.Factory
{
    public class DaprClientFactoryTests
    {
        [Fact]
        public void CreateClient_ReturnsDaprClientWrapperInstance()
        {
            // Arrange
            var factory = new DaprClientFactory();

            // Act
            var wrapper = factory.CreateClient();

            // Assert
            Assert.NotNull(wrapper);
            Assert.IsType<DaprClientWrapper>(wrapper);
        }

        [Fact]
        public async Task DaprClientWrapper_PublishEventAsync_DelegatesToDaprClient()
        {
            // Arrange
            var daprClientMock = new Mock<DaprClient>();
            daprClientMock.Setup(x => x.PublishEventAsync(
                It.IsAny<string>(),
                It.IsAny<string>(),
                It.IsAny<object>(),
                It.IsAny<CancellationToken>())).Returns(Task.CompletedTask);
            var wrapper = new DaprClientWrapper(daprClientMock.Object);

            // Act
            await wrapper.PublishEventAsync("pubsub", "topic", new { foo = "bar" }, CancellationToken.None);

            // Assert
            daprClientMock.Verify(x => x.PublishEventAsync(
                "pubsub", "topic", It.IsAny<object>(), CancellationToken.None), Times.Once);
        }

        [Fact]
        public async Task DaprClientWrapper_InvokeMethodAsync_DelegatesToDaprClient()
        {
            // Arrange
            var daprClientMock = new Mock<DaprClient>();
            daprClientMock.Setup(x => x.InvokeMethodAsync<string>(
                It.IsAny<HttpMethod>(),
                It.IsAny<string>(),
                It.IsAny<string>(),
                It.IsAny<CancellationToken>())).ReturnsAsync("result");
            var wrapper = new DaprClientWrapper(daprClientMock.Object);

            // Act
            var result = await wrapper.InvokeMethodAsync<string>(HttpMethod.Get, "app", "method", CancellationToken.None);

            // Assert
            Assert.Equal("result", result);
            daprClientMock.Verify(x => x.InvokeMethodAsync<string>(
                HttpMethod.Get, "app", "method", CancellationToken.None), Times.Once);
        }
    }
}
