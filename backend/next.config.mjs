import { withKumaUI } from "@kuma-ui/next-plugin"

/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    images: { 
        unoptimized: false,
        domains: ["image.tmdb.org","assets.aceternity.com"]
     } 
};

export default withKumaUI(nextConfig, {
    // The destination to emit an actual CSS file. If not provided, the CSS will be injected via virtual modules.
     outputDir: "./.kuma", // Optional
     // Enable WebAssembly support for Kuma UI. Default is false and will use Babel to transpile the code.
    //  wasm: true // Optional
});