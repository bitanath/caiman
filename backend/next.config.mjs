import { withKumaUI } from "@kuma-ui/next-plugin"

/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    images: { 
        unoptimized: false,
        domains: ["assets.aceternity.com"]
     },
     async headers() {
        return [
            {
                // matching all API routes and setting Authorization Header to allowed
                source: "/api/:path*",
                headers: [
                    { key: "Access-Control-Allow-Credentials", value: "true" },
                    { key: "Access-Control-Allow-Origin", value: "*" }, // replace this your actual origin
                    { key: "Access-Control-Allow-Methods", value: "GET,DELETE,PATCH,POST,PUT" },
                    { key: "Access-Control-Allow-Headers", value: "X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Authorization, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version" },
                ]
            }
        ]
    },
    // webpack: function (config, options) {
    //     config.experiments = { asyncWebAssembly: false, syncWebAssembly: true, layers: true, topLevelAwait: false };
    
    //     return config;
    // } 
};

export default withKumaUI(nextConfig, {
     outputDir: "./.kuma",
});

function patchWasmModuleImport(config, isServer) {
    config.experiments = Object.assign(config.experiments || {}, {
      asyncWebAssembly: true,
    });
    config.module.defaultRules = [
      {
        type: 'javascript/auto',
        resolve: {},
      },
      {
        test: /\.json$/i,
        type: 'json',
      },
    ];
    config.optimization.moduleIds = 'named';
  
    config.module.rules.push({
      test: /\.wasm$/,
      type: 'asset/resource',
    });
  
    // TODO: improve this function -> track https://github.com/vercel/next.js/issues/25852
    if (isServer) {
      config.output.webassemblyModuleFilename = './../static/wasm/[modulehash].wasm';
    } else {
      config.output.webassemblyModuleFilename = 'static/wasm/[modulehash].wasm';
    }
}