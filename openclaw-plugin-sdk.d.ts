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
    registerCommand: (command: {
      name: string;
      description: string;
      acceptsArgs?: boolean;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      handler: (ctx: any) => Promise<{ text: string }>;
    }) => void;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    registerHook: (hookName: string, handler: (event: any) => Promise<void>, opts?: { name?: string }) => void;
    config: {
      agents?: {
        list?: Array<{ id: string; [key: string]: unknown }>;
      };
      [key: string]: unknown;
    };
    runtime: {
      config: {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        loadConfig(): any;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        writeConfigFile(config: any): Promise<void>;
      };
      agent: {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        resolveAgentWorkspaceDir(config: any, agentId: string): string | undefined;
      };
    };
  }
}
