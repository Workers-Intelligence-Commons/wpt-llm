import type { ComponentType } from "react";
import { conversations } from "./conversations";
import { images, recipes, signers } from "./documents";
import { newSource } from "./new-source";
import { ratings } from "./ratings";
import { sources } from "./sources";
import { tasks } from "./tasks";

// One entry per page; its routes are in wpt_web/pages/<name>.py.
export type Page = { path: string; title: string; Component: ComponentType<{ rest: string[] }> };

export const pages: Page[] = [sources, newSource, conversations, tasks, ratings, recipes, images, signers];
