import { initializeApp } from 'firebase/app';
import { clientConfig, serverConfig } from './src/config';

export const app = initializeApp(clientConfig);