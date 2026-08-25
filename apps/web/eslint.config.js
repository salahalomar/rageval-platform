// Flat config (ESLint 9). Deliberately small: TypeScript already catches the class of
// bug a large rule set would, and a hundred stylistic rules in a repository read by
// interviewers is noise pretending to be rigour.
import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import tsParser from "@typescript-eslint/parser";
import tsPlugin from "@typescript-eslint/eslint-plugin";

export default [
  { ignores: ["dist/**", "src/lib/schema.d.ts"] },
  js.configs.recommended,
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      parser: tsParser,
      parserOptions: { ecmaVersion: 2022, sourceType: "module", ecmaFeatures: { jsx: true } },
      globals: {
        document: "readonly",
        window: "readonly",
        fetch: "readonly",
        Response: "readonly",
        TextDecoder: "readonly",
        console: "readonly",
        process: "readonly",
      },
    },
    plugins: { "@typescript-eslint": tsPlugin, "react-hooks": reactHooks },
    rules: {
      ...tsPlugin.configs.recommended.rules,
      ...reactHooks.configs.recommended.rules,
      // `no-undef` duplicates the compiler for TypeScript files and gets JSX types wrong.
      "no-undef": "off",
      "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
    },
  },
];
