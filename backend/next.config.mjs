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
    } 
};

export default withKumaUI(nextConfig, {
    // The destination to emit an actual CSS file. If not provided, the CSS will be injected via virtual modules.
     outputDir: "./.kuma", // Optional
     // Enable WebAssembly support for Kuma UI. Default is false and will use Babel to transpile the code.
    //  wasm: true // Optional
});