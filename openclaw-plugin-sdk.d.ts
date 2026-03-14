declare module "openclaw/plugin-sdk/memory-lancedb" {
  export interface OpenClawPluginApi {
    pluginConfig: unknown;
    logger: {
      info: (...args: unknown[]) => void;
      warn: (...args: unknown[]) => void;
      error: (...args: unknown[]) => void;
      debug: (...args: unknown[]) => void;
    };
    resolvePath: (p: string) => string;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    registerTool: (toolOrFactory: any, opts: { name: string }) => void;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    registerCli: (registrar: any, opts: { commands: string[] }) => void;
    registerService: (service: {
      id: string;
      start: () => void;
      stop: () => void;
    }) => void;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    on: (event: string, handler: (...args: any[]) => Promise<unknown>) => void;
  }
}
