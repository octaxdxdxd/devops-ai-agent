import { expect, test } from "@playwright/test";

test("cockpit renders on desktop and mobile", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("AIOps Cockpit")).toBeVisible();
  await expect(page.getByPlaceholder(/Investigate why/)).toBeVisible();
});
