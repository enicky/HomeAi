FROM --platform=$BUILDPLATFORM  mcr.microsoft.com/dotnet/sdk:8.0 AS build-env
WORKDIR /app

RUN echo "BUILDPLATFORM : $BUILDPLATFORM"
# Copy csproj and restore as distinct layers
COPY DataExporter/*.csproj ./DataExporter/
COPY Common/*.csproj ./Common/

RUN dotnet restore ./DataExporter/DataExporter.csproj

# Copy everything else and build
COPY DataExporter ./DataExporter
COPY Common ./Common
RUN dotnet publish ./DataExporter/DataExporter.csproj -c Release -o out

# Build runtime image
FROM mcr.microsoft.com/dotnet/aspnet:8.0

RUN apt update && \
    apt --no-install-recommends install -y sudo && \
    addgroup --gid 1000 nonroot && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos "" nonroot && \
    echo 'nonroot ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers && \
    rm -rf /var/lib/apt/lists/*

USER nonroot
WORKDIR /app
COPY --chown=nonroot:nonroot --chmod=755  --from=build-env /app/out .
EXPOSE 8080
EXPOSE 80
ENTRYPOINT ["dotnet", "DataExporter.dll"]