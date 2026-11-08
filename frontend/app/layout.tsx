import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DevBrain | Developer Workspace",
  description: "Your personal developer assistant and knowledge base.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
