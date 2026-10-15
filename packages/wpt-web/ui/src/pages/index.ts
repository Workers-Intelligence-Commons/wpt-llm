import type { ComponentType } from "react";
import { sources } from "./sources";

// One entry per page; its routes are in wpt_web/pages/<name>.py.
export type Page = { path: string; title: string; Component: ComponentType<{ rest: string[] }> };

export const pages: Page[] = [sources];
