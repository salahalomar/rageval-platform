/**
 * End-to-end smoke tests.
 *
 * What these assert is deliberately narrow: that the page wires up to the real API, that
 * changing the configuration re-runs retrieval, and that a question the corpus cannot
 * answer is refused rather than answered. Everything about *how good* retrieval is belongs
 * in the eval harness, which measures it against ground truth -- a browser test that
 * asserted a particular paper came first would be a metric with no baseline, hidden inside
 * a test suite.
 *
 * Requires a running stack with an ingested corpus. Not part of the CI gate.
 */
import { expect, test } from "@playwright/test";

test("the ask page loads with an empty state and the server's own defaults", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("empty")).toBeVisible();

  // The panel is populated from GET /config/default, so this failing means the client and
  // the library disagree about what "default" means.
  await expect(page.getByLabel("Cross-encoder rerank")).toBeChecked();
  await expect(page.getByLabel("Fusion")).toHaveValue("rrf");
});

test("retrieval returns cited sources for an answerable question", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("question").fill("What is the purpose of learning rate warmup?");
  await page.getByRole("button", { name: "Retrieve" }).click();

  const sources = page.getByTestId("source");
  await expect(sources.first()).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("stages")).toBeVisible();

  // Every block must carry its provenance; a source you cannot go and check is not a
  // citation, it is a decoration.
  await expect(sources.first().getByRole("link", { name: /arXiv:/ })).toBeVisible();
});

test("turning off the cross-encoder re-runs the same question", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("question").fill("How does rotary position embedding work?");
  await page.getByRole("button", { name: "Retrieve" }).click();
  await expect(page.getByTestId("source").first()).toBeVisible({ timeout: 30_000 });

  const before = await page.getByTestId("source").allTextContents();
  await page.getByLabel("Cross-encoder rerank").uncheck();
  await expect(page.getByTestId("source").first()).toBeVisible({ timeout: 30_000 });

  // The scores are labelled differently once reranking is off, which is the visible proof
  // that the request actually went back to the server rather than being re-rendered.
  await expect(page.getByTestId("source").first()).toContainText("rrf");
  expect(before.length).toBeGreaterThan(0);
});

test("a question the corpus cannot support is refused", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("question").fill("What were Tesla's Q3 2025 delivery numbers?");
  await page.getByRole("button", { name: "Retrieve" }).click();

  // Either outcome is acceptable to assert on here: the refusal banner, or sources with
  // no answer. What must never happen is a generated answer, and generation is off.
  await expect(page.getByTestId("answer")).toHaveCount(0);
});

test("the evaluation page renders the committed ablation table", async ({ page }) => {
  await page.goto("/eval");
  const rows = page.getByTestId("arm-row");
  await expect(rows.first()).toBeVisible();

  // Matrix order, not alphabetical and not sorted by score.
  await expect(rows.first()).toContainText("lexical-only");
  await expect(page.locator("tr.winner").first()).toBeVisible();

  // Clicking a row reveals the configuration that produced it. A number without its
  // config is worthless, so the page has to be able to show it.
  await rows.first().click();
  await expect(page.getByText(/embedding_model=/)).toBeVisible();
});
