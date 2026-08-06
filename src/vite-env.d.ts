/// <reference types="vite/client" />

declare namespace React {
  interface HTMLAttributes<T> extends DOMAttributes<T> {
    'data-tauri-drag-region'?: boolean;
  }
}
