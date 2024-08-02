import { clientConfig } from '@/config'
import { initializeApp } from 'firebase/app'

//client side firestore queries are secured using the user that requested them
export const app = initializeApp(clientConfig)
